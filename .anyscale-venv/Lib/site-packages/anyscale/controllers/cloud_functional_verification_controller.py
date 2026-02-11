import concurrent.futures
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set

from click import ClickException
from rich.console import Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table

from anyscale.cli_logger import LogsLogger
from anyscale.client.openapi_client.models import (
    ApplyProductionServiceV2Model,
    CloudAnalyticsEventName,
    CloudProviders,
    ComputeTemplateQuery,
    CreateExperimentalWorkspace,
    ServiceEventCurrentState,
    SessionState,
)
from anyscale.controllers.base_controller import BaseController
from anyscale.project_utils import get_default_project
from anyscale.sdk.anyscale_client.models import (
    ComputeNodeType,
    CreateClusterCompute,
    CreateClusterComputeConfig,
    ServiceConfig,
)
from anyscale.util import confirm, get_endpoint
from anyscale.utils.cloud_utils import CloudEventProducer


POLL_INTERVAL_SECONDS = 10
WORKSPACE_VERIFICATION_TIMEOUT_MINUTES = 10
SERVICE_VERIFICATION_TIMEOUT_MINUTES = 30  # for a single rollout

# default values for cluster compute config
MAXIMUM_UPTIME_MINUTES = 15
IDLE_TERMINATION_MINUTES = 5
HEAD_NODE_TYPE_AWS = "m5.xlarge"  # on demand price ~$0.20 per hour
HEAD_NODE_TYPE_GCP = "n2-highmem-2"  # on demand price ~$0.13 per hour
CREATE_COMPUTE_CONFIG_TIMEOUT_SECONDS = 600  # 10 minutes

# Workspace verification will fail fast if any of the following logs are found
WORKSPACE_FAIL_FAST_MATCHING_LOGS = ["Failed to execute"]


class CloudFunctionalVerificationType(str, Enum):
    WORKSPACE = "WORKSPACE"
    SERVICE = "SERVICE"


class CloudFunctionalVerificationTask:
    def __init__(self, task_id: TaskID, description: str):
        self.task_id = task_id
        self.description = description
        self.succeeded = False
        self.completed = False

    def update(self, succeeded: bool, description: str, completed: bool = False):
        self.description = description
        self.succeeded = succeeded
        self.completed = completed


class CloudFunctionalVerificationController(BaseController):
    def __init__(
        self,
        cloud_event_producer: CloudEventProducer,
        log: Optional[LogsLogger] = None,
        initialize_auth_api_client: bool = True,
    ):
        if log is None:
            log = LogsLogger()

        super().__init__(initialize_auth_api_client=initialize_auth_api_client)
        self.cloud_event_producer = cloud_event_producer
        self.log = log

        self.event_log_num: Dict[CloudFunctionalVerificationType, int] = {
            CloudFunctionalVerificationType.WORKSPACE: 0,
            CloudFunctionalVerificationType.SERVICE: 0,
        }

        # Used for rich live console
        self.step_progress: Dict[CloudFunctionalVerificationType, Progress] = {}
        self.overall_progress: Dict[CloudFunctionalVerificationType, Progress] = {}
        self.task_ids: Dict[CloudFunctionalVerificationType, TaskID] = {}
        self.event_log_tables: Dict[CloudFunctionalVerificationType, Table] = {}

    @staticmethod
    def get_head_node_type(cloud_provider: CloudProviders) -> str:
        """
        Get the default head node type for the given cloud provider.
        """
        if cloud_provider == CloudProviders.AWS:
            return HEAD_NODE_TYPE_AWS
        elif cloud_provider == CloudProviders.GCP:
            return HEAD_NODE_TYPE_GCP
        raise ClickException(f"Unsupported cloud provider: {cloud_provider}")

    def get_or_create_cluster_compute(
        self, cloud_id: str, cloud_provider: CloudProviders
    ) -> str:
        """
        Get or create a cluster compute for cloud functional verification
        """
        cluster_compute_name = f"functional_verification_{cloud_id}"
        cluster_compute_version = 1

        cluster_computes = self.api_client.search_compute_templates_api_v2_compute_templates_search_post(
            ComputeTemplateQuery(
                orgwide=True,
                name={"equals": cluster_compute_name},
                include_anonymous=True,
                version=cluster_compute_version,
            )
        ).results
        if len(cluster_computes) > 0:
            return cluster_computes[0].id

        head_node_instance_type = self.get_head_node_type(cloud_provider)
        # no cluster compute found, create one
        cluster_compute_config = CreateClusterComputeConfig(
            cloud_id=cloud_id,
            max_workers=0,
            allowed_azs=["any"],
            head_node_type=ComputeNodeType(
                name="head_node_type", instance_type=head_node_instance_type,
            ),
            maximum_uptime_minutes=MAXIMUM_UPTIME_MINUTES,
            idle_termination_minutes=IDLE_TERMINATION_MINUTES,
            worker_node_types=[],
        )
        if cloud_provider == CloudProviders.AWS:
            cluster_compute_config.aws_advanced_configurations_json = {
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "cloud_functional_verification", "Value": cloud_id,}
                        ],
                    }
                ]
            }
        elif cloud_provider == CloudProviders.GCP:
            cluster_compute_config.gcp_advanced_configurations_json = {
                "instance_properties": {
                    "labels": {"cloud_functional_verification": cloud_id},
                }
            }

        # Add retries with exponential backoff here to handle the case where
        # the cloud admin zone is not ready yet.
        start_time = time.time()
        end_time = start_time + CREATE_COMPUTE_CONFIG_TIMEOUT_SECONDS

        delay = 1
        max_delay = 64
        while time.time() < end_time:
            try:
                cluster_compute = self.anyscale_api_client.create_cluster_compute(
                    CreateClusterCompute(
                        name=cluster_compute_name,
                        config=cluster_compute_config,
                        anonymous=True,
                    )
                ).result
                return cluster_compute.id
            except ClickException:
                # Retry if the cloud admin zone is not ready yet
                pass
            delay = min(delay, max_delay)
            # Add jitter to avoid synchronized retries
            jitter = random.uniform(0, delay / 2)
            time.sleep(delay + jitter)
            delay *= 2  # exponential backoff
        raise ClickException(
            "Timed out waiting for compute config to be created. Please try again."
        )

    def _prepare_verification(self, cloud_id: str, cloud_provider: CloudProviders):
        """
        Generate the required parameters for cloud functional verification.
        """

        cluster_env_build_id = self.get_default_cluster_env_build_id()

        project_id = get_default_project(
            self.api_client, self.anyscale_api_client, parent_cloud_id=cloud_id
        ).id

        cluster_compute_id = self.get_or_create_cluster_compute(
            cloud_id, cloud_provider
        )

        return cluster_compute_id, cluster_env_build_id, project_id

    def get_default_cluster_env_build_id(self):
        try:
            cluster_env_list = self.api_client.list_application_templates_api_v2_application_templates_get(
                defaults_first=True
            ).results
            if len(cluster_env_list) == 0:
                raise ClickException("No cluster environments found")
        except Exception as e:  # noqa: BLE001
            raise ClickException(f"Failed to list cluster environments: {e}")

        for cluster_env in cluster_env_list:
            if cluster_env.is_default:
                return cluster_env.latest_build.id
        return cluster_env_list[0].latest_build.id

    @contextmanager
    def _create_task(self, function: CloudFunctionalVerificationType, description: str):
        """
        Create a task on the console for cloud functional verification
        """
        task_id = self.step_progress[function].add_task(description)
        task = CloudFunctionalVerificationTask(task_id, description)
        try:
            yield task
        finally:
            self._update_console(
                task.succeeded, function, task_id, task.description, task.completed
            )

    def create_workspace(self, cloud_id: str, cloud_provider: CloudProviders):
        """
        Create a workspace for cloud functional verification
        """
        (
            cluster_compute_id,
            cluster_env_build_id,
            project_id,
        ) = self._prepare_verification(cloud_id, cloud_provider)

        create_workspace_arg = CreateExperimentalWorkspace(
            name=f"fxnvrf_{cloud_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            description=f"workspace for cloud {cloud_id} functional verification",
            project_id=project_id,
            cloud_id=cloud_id,
            compute_config_id=cluster_compute_id,
            cluster_environment_build_id=cluster_env_build_id,
            idle_timeout_minutes=IDLE_TERMINATION_MINUTES,
        )

        workspace = self.api_client.create_workspace_api_v2_experimental_workspaces_post(
            create_workspace_arg
        ).result

        return workspace

    def verify_workspace(self, cloud_id: str, cloud_provider: CloudProviders) -> bool:
        """
        Verifies that the workspace is setup correctly on the given cloud.
        """
        # Create workspace
        with self._create_task(
            CloudFunctionalVerificationType.WORKSPACE, "Creating workspace..."
        ) as create_workspace_task:
            try:
                workspace = self.create_workspace(cloud_id, cloud_provider)
            except ClickException as e:
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.WORKSPACE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                create_workspace_task.update(
                    False, f"[bold red]Failed to create workspace: {e}"
                )
                return False
            url = get_endpoint(f"/workspaces/{workspace.id}")
            create_workspace_task.update(
                True, f"[bold green]Workspace created at {url}"
            )

        # Wait until workspace is active
        def get_workspace_status(workspace_id):
            return self.api_client.get_workspace_api_v2_experimental_workspaces_workspace_id_get(
                workspace_id
            ).result.state

        allowed_status_set = {
            SessionState.RUNNING,
            SessionState.STARTINGUP,
            SessionState.AWAITINGSTARTUP,
            SessionState.UPDATING,
        }

        with self._create_task(
            CloudFunctionalVerificationType.WORKSPACE,
            "Waiting for workspace to become active...",
        ) as wait_task:
            try:
                self.poll_until_active(
                    CloudFunctionalVerificationType.WORKSPACE,
                    workspace,
                    get_workspace_status,
                    SessionState.RUNNING,
                    allowed_status_set,
                    wait_task,
                    WORKSPACE_VERIFICATION_TIMEOUT_MINUTES,
                )
            except ClickException as e:
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.WORKSPACE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                wait_task.update(
                    False,
                    f"[bold red]Error: {e}. Please click on the URL above to check the logs.",
                )
                return False
            wait_task.update(True, "[bold green]Workspace is active.")

        # terminate workspace
        with self._create_task(
            CloudFunctionalVerificationType.WORKSPACE, "Terminating workspace..."
        ) as terminate_workspace_task:
            try:
                # terminate the cluster leads to workspace termination
                self.anyscale_api_client.terminate_cluster(workspace.cluster_id, {})
                allowed_status_set = {
                    SessionState.TERMINATED,
                    SessionState.TERMINATING,
                    SessionState.RUNNING,
                }
                self.poll_until_active(
                    CloudFunctionalVerificationType.WORKSPACE,
                    workspace,
                    get_workspace_status,
                    SessionState.TERMINATED,
                    allowed_status_set,
                    terminate_workspace_task,
                    WORKSPACE_VERIFICATION_TIMEOUT_MINUTES,
                )
                # archive workspace
                self.api_client.delete_workspace_api_v2_experimental_workspaces_workspace_id_delete(
                    workspace.id
                )
            except ClickException as e:
                terminate_workspace_task.update(
                    False,
                    f"[bold red]Failed to terminate workspace: {e}",
                    completed=True,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.WORKSPACE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                return False
            terminate_workspace_task.update(
                True, "[bold green]Workspace terminated.", completed=True
            )

        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.WORKSPACE_FUNCTIONAL_VERIFIED, succeeded=True,
        )
        return True

    def rollout_service(
        self,
        cloud_id: str,
        cloud_provider: CloudProviders,
        *,
        version: str = "v1",
        service_name: Optional[str] = None,
        canary_percent: Optional[int] = None,
    ):
        """
        Roll out a service for cloud functional verification
        """
        (
            cluster_compute_id,
            cluster_env_build_id,
            project_id,
        ) = self._prepare_verification(cloud_id, cloud_provider)

        service_name = (
            service_name
            or f"fxnvrf_{cloud_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )

        service_config = ApplyProductionServiceV2Model(
            name=service_name,
            description=f"service for cloud {cloud_id} functional verification",
            project_id=project_id,
            ray_serve_config={
                "applications": [
                    {
                        "import_path": "serve_hello:entrypoint",
                        "runtime_env": {
                            "working_dir": "https://github.com/anyscale/docs_examples/archive/refs/heads/main.zip",
                            "env_vars": {
                                "SERVE_RESPONSE_MESSAGE": f"cloud functional verification {version}",
                            },
                        },
                    }
                ],
            },
            version=version,
            build_id=cluster_env_build_id,
            compute_config_id=cluster_compute_id,
            config=ServiceConfig(
                max_uptime_timeout_sec=SERVICE_VERIFICATION_TIMEOUT_MINUTES * 2 * 60,
            ),
            canary_percent=canary_percent,
        )

        # Rollout service
        service = self.api_client.apply_service_api_v2_services_v2_apply_put(
            service_config
        ).result

        return service

    def verify_service(  # noqa: PLR0911
        self, cloud_id: str, cloud_provider: CloudProviders
    ) -> bool:
        """
        Verifies that the service can be deployed and upgraded on the given cloud.
        """
        # Deploy service
        with self._create_task(
            CloudFunctionalVerificationType.SERVICE, "Deploying service..."
        ) as deploy_service_task:
            try:
                service = self.rollout_service(cloud_id, cloud_provider, version="v1")
            except ClickException as e:
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.SERVICE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                deploy_service_task.update(
                    False, f"[bold red]Failed to deploy service: {e}"
                )
                return False
            url = get_endpoint(f"/services/{service.id}")
            deploy_service_task.update(True, f"[bold green]Service deployed at {url}")

        # Wait until service is active
        def get_service_status(service_id):
            return self.api_client.get_service_api_v2_services_v2_service_id_get(
                service_id
            ).result.current_state

        # We add all service states into the set
        allowed_status_set = set(ServiceEventCurrentState.allowable_values)

        with self._create_task(
            CloudFunctionalVerificationType.SERVICE,
            "Waiting for service to become active...",
        ) as wait_task:
            try:
                self.poll_until_active(
                    CloudFunctionalVerificationType.SERVICE,
                    service,
                    get_service_status,
                    ServiceEventCurrentState.RUNNING,
                    allowed_status_set,
                    wait_task,
                    SERVICE_VERIFICATION_TIMEOUT_MINUTES,
                )
            except ClickException as e:
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.SERVICE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                wait_task.update(
                    False,
                    f"[bold red]Error: {e}. Please click on the URL above to check the logs.",
                )
                return False
            wait_task.update(True, "[bold green]Service is active.")

        # Upgrade service
        with self._create_task(
            CloudFunctionalVerificationType.SERVICE, "Upgrading service..."
        ) as upgrade_service_task:
            try:
                service = self.rollout_service(
                    cloud_id,
                    cloud_provider,
                    version="v2",
                    service_name=service.name,
                    canary_percent=100,
                )
            except ClickException as e:
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.SERVICE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                upgrade_service_task.update(
                    False, f"[bold red]Failed to upgrade service: {e}"
                )
                return False

            try:
                self.poll_until_active(
                    CloudFunctionalVerificationType.SERVICE,
                    service,
                    get_service_status,
                    ServiceEventCurrentState.RUNNING,
                    allowed_status_set,
                    upgrade_service_task,
                    SERVICE_VERIFICATION_TIMEOUT_MINUTES,
                )
            except ClickException as e:
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.SERVICE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                upgrade_service_task.update(
                    False,
                    f"[bold red]Error: {e}. Please click on the URL above to check the logs.",
                )
                return False
            upgrade_service_task.update(
                True, "[bold green]Service upgraded successfully."
            )

        # Terminate service
        with self._create_task(
            CloudFunctionalVerificationType.SERVICE, "Terminating service..."
        ) as terminate_service_task:
            try:
                self.api_client.terminate_service_api_v2_services_v2_service_id_terminate_post(
                    service.id
                )
                self.poll_until_active(
                    CloudFunctionalVerificationType.SERVICE,
                    service,
                    get_service_status,
                    ServiceEventCurrentState.TERMINATED,
                    allowed_status_set,
                    terminate_service_task,
                    SERVICE_VERIFICATION_TIMEOUT_MINUTES,
                )
            except ClickException as e:
                terminate_service_task.update(
                    False,
                    f"[bold red]Failed to terminate service: {e}",
                    completed=True,
                )
                self.cloud_event_producer.produce(
                    CloudAnalyticsEventName.SERVICE_FUNCTIONAL_VERIFIED,
                    succeeded=False,
                    internal_error=str(e),
                )
                return False
            terminate_service_task.update(
                True, "[bold green]Service terminated.", completed=True
            )

        self.cloud_event_producer.produce(
            CloudAnalyticsEventName.SERVICE_FUNCTIONAL_VERIFIED, succeeded=True,
        )
        return True

    def poll_until_active(  # noqa: PLR0913
        self,
        function_type: CloudFunctionalVerificationType,
        function: Any,
        get_current_status: Callable[[str], Any],
        goal_status: Any,
        allowed_status_set: Set[Any],
        wait_task: CloudFunctionalVerificationTask,
        timeout_minutes: int,
    ) -> bool:
        """
        Polling until it is active.
        """
        start_time = time.time()
        end_time = start_time + timeout_minutes * 60
        while time.time() < end_time:
            time.sleep(POLL_INTERVAL_SECONDS)
            try:
                current_status = get_current_status(function.id)
                succeeded = self._render_event_log(function_type, function)
                if not succeeded:
                    raise ClickException(
                        f"{function_type.capitalize()} verification failed! Please check the logs and terminate the {function_type.lower()} manually."
                    )
            except ClickException as e:
                raise ClickException(
                    f"Failed to get {function_type.lower()} status: {e}"
                ) from None
            self._update_task_in_step_progress(
                function_type,
                wait_task.task_id,
                f"{wait_task.description} [{time.strftime('%H:%M:%S', time.localtime())}] Current status: {current_status}",
            )
            if current_status == goal_status:
                return True
            if current_status not in allowed_status_set:
                raise ClickException(
                    f"{function_type.capitalize()} is in an unexpected state: {current_status}"
                )
        raise ClickException(
            f"Timed out waiting for {function_type.lower()} to become active"
        )

    def verify(
        self,
        function: CloudFunctionalVerificationType,
        cloud_id: str,
        cloud_provider: CloudProviders,
    ) -> bool:
        """
        Kick off a single functional verification given the function type.
        """
        if function == CloudFunctionalVerificationType.WORKSPACE:
            return self.verify_workspace(cloud_id, cloud_provider)
        elif function == CloudFunctionalVerificationType.SERVICE:
            return self.verify_service(cloud_id, cloud_provider)
        return False

    def _update_console(
        self,
        succeeded: bool,
        function: CloudFunctionalVerificationType,
        task_id: TaskID,
        description: str,
        completed: bool = False,
    ):
        """
        Update the console based on the verification result
        """
        self._update_overall_progress(succeeded, function, completed)
        self._finish_task_in_step_progress(function, task_id, description)

    def _update_task_in_step_progress(
        self,
        function: CloudFunctionalVerificationType,
        task_id: TaskID,
        description: str,
    ) -> None:
        """
        Update the task description in step progress
        """
        self.step_progress[function].update(task_id, description=description)

    def _finish_task_in_step_progress(
        self,
        function: CloudFunctionalVerificationType,
        task_id: TaskID,
        description: str,
    ) -> None:
        """
        Finish a task in step progress and update the description
        """
        self.step_progress[function].stop_task(task_id)
        self.step_progress[function].update(task_id, description=description)

    def _update_overall_progress(
        self,
        verification_result: bool,
        function: CloudFunctionalVerificationType,
        completed: bool = False,
    ) -> None:
        """
        Update overall progress based on the verification result
        """
        if verification_result:
            self.overall_progress[function].advance(self.task_ids[function], 1)
            if completed:
                self.overall_progress[function].update(
                    self.task_ids[function],
                    description=f"[bold green]{function.capitalize()} verification succeeded!",
                )
        else:
            self.overall_progress[function].stop_task(self.task_ids[function])
            self.overall_progress[function].update(
                self.task_ids[function],
                description=f"[bold red]{function.capitalize()} verification failed. Please check the logs and terminate the {function.lower()} manually.",
            )

    def _format_service_event_log(self, event_log: Any) -> str:
        description = event_log.message.replace("\n", ". ") if event_log.message else ""
        if event_log.event_type:
            return f"[bold cyan]{event_log.created_at}[/bold cyan] [bold blue]{event_log.event_type}[/bold blue] [bold green]{description}[/bold green]"
        else:
            return f"[bold cyan]{event_log.created_at}[/bold cyan] [bold green]{description}[/bold green]"

    def _render_event_log(
        self, function_type: CloudFunctionalVerificationType, function: Any
    ) -> bool:
        if function_type == CloudFunctionalVerificationType.WORKSPACE:
            workspace_log = self.api_client.get_startup_logs_api_v2_sessions_session_id_startup_logs_get(
                function.cluster_id, self.event_log_num[function_type], 100000000
            ).result
            if workspace_log.num_lines == self.event_log_num[function_type]:
                # no new logs
                return True

            # Output logs
            self.event_log_tables[function_type].add_row(escape(workspace_log.lines))
            self.event_log_num[function_type] = workspace_log.num_lines

            # fail fast if any of the following logs are found
            return not any(
                failure_log in workspace_log.lines
                for failure_log in WORKSPACE_FAIL_FAST_MATCHING_LOGS
            )
        elif function_type == CloudFunctionalVerificationType.SERVICE:
            service_events = self.api_client.get_service_events_api_v2_services_v2_service_id_events_get(
                function.id
            ).results
            starting_pos = len(service_events) - self.event_log_num[function_type] - 1
            for idx in range(starting_pos, -1, -1):
                self.event_log_tables[function_type].add_row(
                    self._format_service_event_log(service_events[idx])
                )
            self.event_log_num[function_type] = len(service_events)
            return True
        else:
            # we should never enter this branch
            return False

    def get_live_console(
        self, functions_to_verify: List[CloudFunctionalVerificationType]
    ) -> Live:
        """
        Get a live console for cloud functional verification.

        Each functional verification contains 3 components:
        1. step progress panel: shows the progress bars for each functional verification step
        2. event log table: event logs for each functional verification
        3. overall progress: overall progress bar for the functional verification
        """
        progress_group = []
        steps = {
            CloudFunctionalVerificationType.WORKSPACE: 3,
            CloudFunctionalVerificationType.SERVICE: 4,
        }
        for function in functions_to_verify:
            progress_table = Table.grid(expand=True)
            step_progress = Progress(
                TextColumn(
                    "{task.description}",
                    table_column=Column(no_wrap=False, overflow="fold"),
                ),
            )
            self.step_progress[function] = step_progress
            event_log_table = Table(box=None)
            self.event_log_tables[function] = event_log_table
            progress_table.add_row(
                Panel(step_progress, title=f"{function.lower()} verification")
            )
            progress_table.add_row(
                Panel(event_log_table, title=f"{function.lower()} event logs")
            )

            progress_group.append(progress_table)

            overall_progress = Progress(
                TimeElapsedColumn(), BarColumn(), TextColumn("{task.description}")
            )
            self.overall_progress[function] = overall_progress
            progress_group.append(overall_progress)
            task_id = overall_progress.add_task("", total=steps[function])
            overall_progress.update(
                task_id, description=f"[bold #AAAAAA]Verifying {function.lower()}..."
            )
            self.task_ids[function] = task_id
        return Live(Group(*progress_group))

    def start_verification(
        self,
        cloud_id: str,
        cloud_provider: CloudProviders,
        functions_to_verify: List[CloudFunctionalVerificationType],
        yes: bool = False,
    ) -> bool:
        """
        Starts cloud functional verification
        """
        with self.log.spinner("Starting functional verification..."):
            self._prepare_verification(cloud_id, cloud_provider)

        self.log.info(
            f"Functional verification for {', '.join(functions_to_verify)} is about to begin."
        )

        confirmation_message = [
            f"It will spin up one {self.get_head_node_type(cloud_provider)} instance for each function.",
        ]

        if CloudFunctionalVerificationType.WORKSPACE in functions_to_verify:
            confirmation_message.append(
                "Workspace verification takes about 5 minutes. "
                "The approximate cost from the cloud provider is about $0.02 (it varies from cloud provider and region)."
            )

        if CloudFunctionalVerificationType.SERVICE in functions_to_verify:
            service_time_estimation = {
                CloudProviders.AWS: 8,
                CloudProviders.GCP: 20,
            }
            confirmation_message.append(
                f"Service verification takes about {service_time_estimation[cloud_provider]} minutes. "
                "The approximate cost from the cloud provider is about $0.10 (it varies from cloud provider and region)."
            )

        confirmation_message.append(
            "The instances will be terminated after verification."
        )

        self.log.info("\n".join(confirmation_message))

        confirm(
            "Continue?", yes,
        )

        verification_results: List[bool] = []
        with self.get_live_console(
            functions_to_verify
        ), concurrent.futures.ThreadPoolExecutor(
            max_workers=len(CloudFunctionalVerificationType)
        ) as executor:
            futures = {
                executor.submit(
                    self.verify, function, cloud_id, cloud_provider
                ): function
                for function in functions_to_verify
            }

            # Wait for all verifications to complete
            for future in concurrent.futures.as_completed(futures):
                function = futures[future]
                try:
                    verification_result = future.result()
                    verification_results.append(verification_result)
                except Exception as e:  # noqa: BLE001
                    self._update_console(
                        False,
                        function,
                        self.task_ids[function],
                        f"[bold red]Failed to verify {function.lower()}: {e}",
                        completed=True,
                    )
                    verification_results.append(False)
        return all(verification_results)
