from __future__ import annotations

import asyncio
import os
import random
import string
import time
from typing import Any, Dict, List, Optional

import click
import tabulate
import yaml

from anyscale.anyscale_pydantic import BaseModel
from anyscale.api_utils.job_logs_util import _get_job_logs_from_storage_bucket_streaming
from anyscale.cli_logger import LogsLogger
from anyscale.client.openapi_client import (
    ComputeTemplate,
    ProductionJob,
    ProductionJobConfig,
)
from anyscale.client.openapi_client.models.create_internal_production_job import (
    CreateInternalProductionJob,
)
from anyscale.client.openapi_client.models.create_job_queue_config import (
    CreateJobQueueConfig,
)
from anyscale.client.openapi_client.models.decorated_production_job import (
    DecoratedProductionJob,
)
from anyscale.client.openapi_client.models.ha_job_states import HaJobStates
from anyscale.commands.util import flatten_tag_dict_to_api_list
from anyscale.controllers.base_controller import BaseController
from anyscale.models.job_model import JobConfig
from anyscale.project_utils import infer_project_id
from anyscale.sdk.anyscale_client.models.job import Job
from anyscale.sdk.anyscale_client.models.jobs_query import JobsQuery
from anyscale.sdk.anyscale_client.models.jobs_sort_field import JobsSortField
from anyscale.sdk.anyscale_client.models.page_query import PageQuery
from anyscale.sdk.anyscale_client.models.sort_by_clause_jobs_sort_field import (
    SortByClauseJobsSortField,
)
from anyscale.sdk.anyscale_client.models.sort_order import SortOrder
from anyscale.util import (
    get_endpoint,
    is_anyscale_workspace,
    populate_unspecified_cluster_configs_from_current_workspace,
    validate_job_config_dict,
)
from anyscale.utils.connect_helpers import search_entities
from anyscale.utils.runtime_env import override_runtime_env_config
from anyscale.utils.workload_types import Workload


log = LogsLogger()

_TERMINAL_STATES = {
    HaJobStates.SUCCESS,
    HaJobStates.TERMINATED,
    HaJobStates.BROKEN,
    HaJobStates.OUT_OF_RETRIES,
}
_PENDING_STATES = {
    HaJobStates.PENDING,
    HaJobStates.AWAITING_CLUSTER_START,
    HaJobStates.ERRORED,
    HaJobStates.RESTARTING,
}

DEFAULT_PAGE_LIMIT = 500


class MiniJobRun(BaseModel):
    last_job_run_id: str
    job_state: str
    error: Optional[str]


class JobController(BaseController):
    def __init__(
        self,
        log: Optional[LogsLogger] = None,
        initialize_auth_api_client: bool = True,
        raise_structured_exception: bool = False,
        auth_token: Optional[str] = None,
    ):
        if log is None:
            log = LogsLogger()

        super().__init__(
            initialize_auth_api_client, raise_structured_exception, cli_token=auth_token
        )
        self.log = log
        self.log.open_block("Output")
        self._cluster_journal_events_start_line = 0

    def submit(
        self,
        job_config_file: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_entrypoint_cmd: Optional[bool] = False,
        entrypoint: Optional[List[str]] = None,
    ) -> str:
        entrypoint = entrypoint or []
        workspace_id = os.environ.get("ANYSCALE_EXPERIMENTAL_WORKSPACE_ID", None)
        if is_anyscale_workspace() and is_entrypoint_cmd:
            entrypoint = [job_config_file, *entrypoint]
            config = self.generate_config_from_entrypoint(
                entrypoint, name, description, workspace_id
            )
            id = self.submit_from_config(config)  # noqa: A001
        elif len(entrypoint) == 0:
            # Assume that job_config_file is a file and submit it.
            config = self.generate_config_from_file(
                job_config_file,
                name=name,
                description=description,
                workspace_id=workspace_id,
            )
            id = self.submit_from_config(config)  # noqa: A001
        elif len(entrypoint) != 0:
            msg = (
                "Within an Anyscale Workspace, `anyscale job submit` takes either a file, or a command. To submit a command, use `anyscale job submit -- my command`."
                if is_anyscale_workspace()
                else "`anyscale job submit` takes one argument, a YAML file configuration. Please use `anyscale job submit my_file`."
            )
            raise click.ClickException(msg)
        return id

    def generate_config_from_entrypoint(
        self,
        entrypoint: List[str],
        name: Optional[str],
        description: Optional[str],
        workspace_id: Optional[str] = None,
    ) -> JobConfig:
        config_dict = {
            "entrypoint": " ".join(entrypoint),
            "name": name,
            "description": description,
            "workspace_id": workspace_id,
        }

        config_dict = populate_unspecified_cluster_configs_from_current_workspace(
            config_dict, self.anyscale_api_client,
        )

        return JobConfig.parse_obj(config_dict)

    def generate_config_from_file(
        self,
        job_config_file: str,
        name: Optional[str],
        description: Optional[str],
        workspace_id: Optional[str] = None,
    ) -> JobConfig:
        config_dict = self._load_config_dict_from_file(job_config_file)
        config_dict["workspace_id"] = workspace_id
        validate_job_config_dict(config_dict, self.api_client)
        config_dict = populate_unspecified_cluster_configs_from_current_workspace(
            config_dict, self.anyscale_api_client
        )
        job_config = JobConfig.parse_obj(config_dict)

        if name:
            job_config.name = name

        if description:
            job_config.description = description

        return job_config

    def submit_from_config(self, job_config: JobConfig):
        # If project id is not specified, try to infer it
        project_id = infer_project_id(
            self.anyscale_api_client,
            self.api_client,
            self.log,
            project_id=job_config.project_id,
            cluster_compute_id=job_config.compute_config_id,
            cluster_compute=job_config.compute_config,
            cloud=job_config.cloud,
        )

        job_config.runtime_env = override_runtime_env_config(
            runtime_env=job_config.runtime_env,
            anyscale_api_client=self.anyscale_api_client,
            api_client=self.api_client,
            workload_type=Workload.JOBS,
            compute_config_id=job_config.compute_config_id,
            log=self.log,
        )
        config_object = ProductionJobConfig(
            entrypoint=job_config.entrypoint,
            runtime_env=job_config.runtime_env,
            build_id=job_config.build_id,
            compute_config_id=job_config.compute_config_id,
            max_retries=job_config.max_retries,
        )

        job_queue_config = (
            CreateJobQueueConfig(**job_config.job_queue_config)
            if job_config.job_queue_config
            else None
        )

        job = self.api_client.create_job_api_v2_decorated_ha_jobs_create_post(
            CreateInternalProductionJob(
                name=job_config.name or self._generate_random_job_name(),
                description=job_config.description or "Job submitted from CLI",
                project_id=project_id,
                workspace_id=job_config.workspace_id,
                config=config_object,
                job_queue_config=job_queue_config,
                tags=job_config.tags,
            )
        ).result
        self.log.info(
            f"Maximum uptime is {self._get_maximum_uptime_output(job)} for clusters launched by this job."
        )
        self.log.info(
            f"Job {job.id} has been successfully submitted. Current state of job: {job.state.current_state}."
        )
        self.log.info(
            f"Query the status of the job with `anyscale job list --job-id {job.id}`."
        )
        self.log.info(
            f"Get the logs for the job with `anyscale job logs --job-id {job.id} --follow`."
        )
        self.log.info(f'View the job in the UI at {get_endpoint(f"/jobs/{job.id}")}')
        return job.id

    def _get_maximum_uptime_output(self, job: ProductionJob) -> str:
        compute_config: ComputeTemplate = self.api_client.get_compute_template_api_v2_compute_templates_template_id_get(
            job.config.compute_config_id
        ).result
        maximum_uptime_minutes = compute_config.config.maximum_uptime_minutes
        if maximum_uptime_minutes and maximum_uptime_minutes > 0:
            return f"set to {maximum_uptime_minutes} minutes"
        return "disabled"

    def _load_config_dict_from_file(self, job_config_file: str) -> Dict[str, Any]:
        if not os.path.exists(job_config_file):
            raise click.ClickException(f"Config file {job_config_file} not found.")

        with open(job_config_file) as f:
            config_dict = yaml.safe_load(f)
        return config_dict

    def resolve_job_id(self, job_id: Optional[str], job_name: Optional[str]) -> str:
        """Resolve and return a job's ID from either an explicit ID or a name.

        Raises click.ClickException if neither is provided or if the name is ambiguous.
        """
        return self._resolve_job_object(job_id, job_name).id

    def list(  # noqa: PLR0913
        self,
        include_all_users: bool,
        name: Optional[str],
        job_id: Optional[str],
        project_id: Optional[str],
        include_archived: bool,
        max_items: Optional[int] = 10,
        states: Optional[List[str]] = None,
        tags: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        This function will list jobs.
        """
        print(f'View your Jobs in the UI at {get_endpoint("/jobs")}')

        jobs_list = []
        if job_id:
            job = self.api_client.get_job_api_v2_decorated_ha_jobs_production_job_id_get(
                job_id
            ).result
            if job.is_service:
                # `job_id` belongs to a service, but this function should list jobs.
                raise click.ClickException(
                    f"ID {job_id} belongs to a Anyscale service. Please get information about "
                    f"this service with `anyscale service list --service-id {job_id}`."
                )

            output_map = {
                "Name": job.name,
                "Id": job.id,
                "Project name": job.project.name,
                "Cluster name": job.last_job_run.cluster.name
                if job.last_job_run and job.last_job_run.cluster
                else None,
                "Current state": job.state.current_state,
                "Creator": job.creator.email,
                "Entrypoint": job.config.entrypoint
                if len(job.config.entrypoint) < 100
                else job.config.entrypoint[:100] + " ...",
            }
            output_str = "\n".join(
                [f"\t{key}: {output_map[key]}" for key in output_map]
            )
            print(output_str)
            return
        else:
            if not include_all_users:
                creator_id = (
                    self.api_client.get_user_info_api_v2_userinfo_get().result.id
                )
            else:
                creator_id = None

            resp = self.api_client.list_decorated_jobs_api_v2_decorated_ha_jobs_get(
                project_id=project_id,
                name=name,
                creator_id=creator_id,
                type_filter="BATCH_JOB",
                archive_status="ALL" if include_archived else "NOT_ARCHIVED",
                count=DEFAULT_PAGE_LIMIT,
                state_filter=states,
                tag_filter=flatten_tag_dict_to_api_list(tags),
            )
            jobs_list.extend(resp.results)
            paging_token = resp.metadata.next_paging_token
            has_more = (paging_token is not None) and (
                max_items is None or len(jobs_list) < max_items
            )
            while has_more:
                resp = self.api_client.list_decorated_jobs_api_v2_decorated_ha_jobs_get(
                    project_id=project_id,
                    name=name,
                    creator_id=creator_id,
                    type_filter="BATCH_JOB",
                    archive_status="ALL" if include_archived else "NOT_ARCHIVED",
                    count=DEFAULT_PAGE_LIMIT,
                    paging_token=paging_token,
                    state_filter=states,
                    tag_filter=flatten_tag_dict_to_api_list(tags),
                )
                jobs_list.extend(resp.results)
                paging_token = resp.metadata.next_paging_token
                has_more = (paging_token is not None) and (
                    max_items is None or len(jobs_list) < max_items
                )
            jobs_list = jobs_list[:max_items]

        jobs_table = [
            [
                job.name,
                job.id,
                job.project.name,
                job.last_job_run.cluster.name
                if job.last_job_run and job.last_job_run.cluster
                else None,
                job.state.current_state,
                job.creator.email,
                job.config.entrypoint
                if len(job.config.entrypoint) < 50
                else job.config.entrypoint[:50] + " ...",
            ]
            for job in jobs_list
        ]

        table = tabulate.tabulate(
            jobs_table,
            headers=[
                "NAME",
                "ID",
                "PROJECT NAME",
                "CLUSTER NAME",
                "CURRENT STATE",
                "CREATOR",
                "ENTRYPOINT",
            ],
            tablefmt="plain",
        )
        print(f"JOBS:\n{table}")

    def archive(self, job_id: Optional[str], job_name: Optional[str]) -> None:
        """
        This function will archive jobs.
        """
        job_resp: DecoratedProductionJob = self._resolve_job_object(job_id, job_name)
        self.api_client.archive_job_api_v2_decorated_ha_jobs_production_job_id_archive_post(
            job_resp.id
        )
        self.log.info(f"Job {job_resp.id} is successfully archived.")

    def terminate(self, job_id: Optional[str], job_name: Optional[str]) -> None:
        """
        This function will terminate jobs.
        """
        job_resp: DecoratedProductionJob = self._resolve_job_object(job_id, job_name)
        job = self.api_client.terminate_job_api_v2_decorated_ha_jobs_production_job_id_terminate_post(
            job_resp.id
        ).result
        self.log.info(f"Job {job.id} has begun terminating...")
        self.log.info(
            f" Current state of Job: {job.state.current_state}. Goal state of Job: {job.state.goal_state}"
        )
        self.log.info(
            f"Query the status of the Job with `anyscale job list --job-id {job.id}`."
        )

    def _resolve_job_object(
        self, job_id: Optional[str], job_name: Optional[str]
    ) -> DecoratedProductionJob:
        """Given job_id or job_name, retrieve decorated ha job spec"""
        if job_id is None and job_name is None:
            raise click.ClickException(
                "Either `--id` or `--name` must be passed in for Job."
            )
        if job_id:
            return self._get_job(job_id)

        jobs_list_resp: List[
            DecoratedProductionJob
        ] = self.api_client.list_decorated_jobs_api_v2_decorated_ha_jobs_get(
            name=job_name, type_filter="BATCH_JOB",
        ).results
        if len(jobs_list_resp) == 0:
            raise click.ClickException(
                f"No Job found with name {job_name}. Please either pass `--id` or list the "
                f"available Jobs with `anyscale job list`."
            )
        if len(jobs_list_resp) > 1:
            raise click.ClickException(
                f"Multiple Jobs found with name {job_name}. Please specify the `--id` instead."
            )
        return jobs_list_resp[0]

    def _get_job(self, job_id: str) -> DecoratedProductionJob:
        job_object = self.api_client.get_job_api_v2_decorated_ha_jobs_production_job_id_get(
            production_job_id=job_id
        ).result
        return job_object

    def _get_formatted_latest_job_run(
        self, job: DecoratedProductionJob
    ) -> Optional[MiniJobRun]:
        job_state = job.state.current_state
        last_job_run_id = job.last_job_run_id
        if job_state in _TERMINAL_STATES and last_job_run_id is None:
            raise click.ClickException(
                f"Can't find latest job run for {job_state} job."
            )
        if not last_job_run_id:
            return None
        return MiniJobRun(
            last_job_run_id=last_job_run_id, job_state=job_state, error=job.state.error
        )

    def _wait_for_a_job_run(
        self, job_id: Optional[str], job_name: Optional[str]
    ) -> DecoratedProductionJob:
        """Waits until the job has a job run and is in a non-pending state
        (encountered when initially submitting).

        Returns the job."""
        start = time.monotonic()
        job = self._resolve_job_object(job_id, job_name)
        last_job_run_id: str = job.last_job_run_id
        job_state = job.state.current_state
        with self.log.spinner("Waiting for job run..."):
            while last_job_run_id is None or job_state in _PENDING_STATES:
                if job_state in _TERMINAL_STATES:
                    raise click.ClickException(
                        f"Can't find latest job run for {job_state} job."
                    )
                time.sleep(max(0, 3 - (time.monotonic() - start)))
                start = time.monotonic()
                job = self._get_job(job.id)
                last_job_run_id = job.last_job_run_id
                if job.state.cluster_id:
                    # Once we have a cluster id, we start printing the cluster journal event logs with each loop.
                    cluster_journal_events = self._api_client.get_startup_logs_api_v2_sessions_session_id_startup_logs_get(
                        job.state.cluster_id,
                        start_line=self._cluster_journal_events_start_line,
                        end_line=10000000,
                    ).result
                    lines = cluster_journal_events.lines.splitlines()
                    if len(lines):
                        self.log.info("Job run found. Cluster launching...")
                    self._cluster_journal_events_start_line = (
                        cluster_journal_events.start_line
                        + cluster_journal_events.num_lines
                    )

                    for line in lines:
                        self.log.info(line)

                job_state = job.state.current_state
        return job

    def _print_logs_using_streaming_job_logs(
        self,
        job_run_id: str,
        follow: bool = False,
        no_cluster_journal_events: bool = False,
    ) -> None:
        """Retrieves logs from the streaming job logs folder in S3/GCS"""
        logs = ""
        next_page_token = None
        # Keep track of if any logs have been printed yet,
        # If not, we assume the cluster recently launched and we
        # give more leeway for the logs to be available.
        have_jobs_started_producing_logs = False
        with self.log.spinner("Retrieving logs..."):
            (
                logs,
                next_page_token,
                cluster_journal_events_start_line,
                job_finished,
            ) = asyncio.run(
                _get_job_logs_from_storage_bucket_streaming(
                    self._api_client,
                    self.log,
                    job_run_id=job_run_id,
                    remove_escape_chars=False,
                    next_page_token=next_page_token,
                    cluster_journal_events_start_line=self._cluster_journal_events_start_line,
                    no_cluster_journal_events=no_cluster_journal_events,
                )
            )

        if len(logs):
            have_jobs_started_producing_logs = True

        for line in logs.splitlines():
            self.log.log(line)

        # Continuously poll for logs.
        terminal_state_count = 0
        while follow:
            start = time.monotonic()
            if job_finished:
                # Let's wait before terminating the loop to give the logs a chance to catch up.
                terminal_state_count += 1
                # Once jobs produce logs, we know we do not have to wait as long for those logs
                # to be uploaded to S3 since the cluster is already running.
                max_iterations = 4 if have_jobs_started_producing_logs else 6
                if terminal_state_count >= max_iterations:
                    self.log.info("Job run reached terminal state.")
                    return

            (
                logs,
                next_page_token,
                cluster_journal_events_start_line,
                job_finished,
            ) = asyncio.run(
                _get_job_logs_from_storage_bucket_streaming(
                    self._api_client,
                    self.log,
                    job_run_id=job_run_id,
                    remove_escape_chars=False,
                    next_page_token=next_page_token,
                    cluster_journal_events_start_line=cluster_journal_events_start_line,
                    no_cluster_journal_events=no_cluster_journal_events,
                )
            )

            for line in logs.splitlines():
                self.log.log(line)

            # Wait at least 5 seconds between iterations.
            time.sleep(max(0, 5 - (time.monotonic() - start)))

    def logs(
        self,
        job_id: Optional[str] = None,
        job_name: Optional[str] = None,
        should_follow: bool = False,
        all_attempts: bool = False,
        no_cluster_journal_events: bool = False,
    ) -> None:
        """
        Fetches logs for a production job using the streaming log S3 bucket.
        Params:
        - no_cluster_journal_events: Controls whether we print out cluster journal logs. (Normally printed to stderr)
        """
        job = self._wait_for_a_job_run(job_id, job_name)
        last_job_run_id: str = job.last_job_run_id
        job_run_ids: List[str] = (
            [last_job_run_id]
            if not all_attempts
            else [job_run.id for job_run in self._get_all_job_runs(job.id)]
        )  # type: ignore
        for job_run_id in job_run_ids:
            self.log.open_block(
                job_run_id, f"Job Run Id: {job_run_id}", auto_close=True,
            )
            self._print_logs_using_streaming_job_logs(
                job_run_id,
                follow=should_follow and job_run_id == last_job_run_id,
                no_cluster_journal_events=no_cluster_journal_events,
            )

    def _generate_random_job_name(self) -> str:
        """Generates a random job name
        Format:
            cli-job-{10 random characters}
        """
        random_chars = "".join(random.choices(string.ascii_letters, k=10)).lower()
        return "cli-job-" + random_chars

    def _get_all_job_runs(self, job_id: str) -> List[Job]:
        """Returns all job runs for a given job id.
        Returned in ascending order by creation time."""
        job_runs: List[Job] = search_entities(
            self.anyscale_api_client.search_jobs,
            JobsQuery(
                ha_job_id=job_id,
                show_ray_client_runs_only=False,
                sort_by_clauses=[
                    SortByClauseJobsSortField(
                        sort_field=JobsSortField.CREATED_AT, sort_order=SortOrder.ASC,
                    )
                ],
                paging=PageQuery(),
            ),
        )
        return job_runs

    def get_authenticated_user_id(self) -> str:
        user_info_response = self.api_client.get_user_info_api_v2_userinfo_get()
        return user_info_response.result.id
