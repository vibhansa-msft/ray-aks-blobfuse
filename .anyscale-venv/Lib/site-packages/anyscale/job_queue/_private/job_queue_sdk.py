from typing import Dict, List, Optional

from anyscale._private.models.model_base import ResultIterator
from anyscale._private.workload import WorkloadSDK
from anyscale.client.openapi_client.models.decorated_job_queue import DecoratedJobQueue
from anyscale.client.openapi_client.models.decoratedjobqueue_list_response import (
    DecoratedjobqueueListResponse,
)
from anyscale.client.openapi_client.models.job_queue_sort_directive import (
    JobQueueSortDirective,
)
from anyscale.client.openapi_client.models.list_response_metadata import (
    ListResponseMetadata,
)
from anyscale.client.openapi_client.models.resource_tag_resource_type import (
    ResourceTagResourceType,
)
from anyscale.client.openapi_client.models.session_state import SessionState
from anyscale.job_queue.models import JobQueueStatus


class PrivateJobQueueSDK(WorkloadSDK):
    """Internal SDK logic for Job Queue operations."""

    def list(  # noqa: PLR0913
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        creator_id: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        cluster_status: Optional[SessionState] = None,
        tags_filter: Optional[Dict[str, List[str]]] = None,
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        sorting_directives: Optional[List[JobQueueSortDirective]] = None,
        include_archived: bool = False,
    ) -> ResultIterator[JobQueueStatus]:
        """List job queues based on specified filters and pagination.

        If job_queue_id is provided, fetches only that specific job queue.
        The include_archived flag is ignored when using job_queue_id.
        """

        if job_queue_id is not None:
            raw = self._resolve_to_job_queue_model(job_queue_id=job_queue_id)

            def _fetch_single_page(
                _token: Optional[str],
            ) -> DecoratedjobqueueListResponse:
                # Only return data on the first call (token=None), simulate single-item page
                if _token is None and raw is not None:
                    results = [raw]
                    metadata = ListResponseMetadata(total=1, next_paging_token=None)
                else:
                    results = []
                    metadata = ListResponseMetadata(total=0, next_paging_token=None)

                return DecoratedjobqueueListResponse(
                    results=results, metadata=metadata,
                )

            return ResultIterator(
                page_token=None,
                max_items=1,  # Return the single fetched item
                fetch_page=_fetch_single_page,
                parse_fn=_parse_decorated_jq_to_status,
            )

        def _fetch_page(token: Optional[str]) -> DecoratedjobqueueListResponse:
            return self.client.list_job_queues(
                name=name,
                creator_id=creator_id,
                cloud=cloud,
                project=project,
                cluster_status=cluster_status,
                tags_filter=tags_filter,
                count=page_size,
                paging_token=token,
                sorting_directives=sorting_directives,
                include_archived=include_archived,
            )

        return ResultIterator(
            page_token=None,
            max_items=max_items,
            fetch_page=_fetch_page,
            parse_fn=_parse_decorated_jq_to_status,
        )

    def status(
        self,
        job_queue_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> JobQueueStatus:
        """Get the status and details for a specific job queue.

        Args:
            job_queue_id: The ID of the job queue.
            name: The name of the job queue (alternative to job_queue_id).
            project: The project name to filter by when using name.
            cloud: The cloud name to filter by when using name.
            include_archived: If True, include archived job queues when searching
                by name. Ignored when using job_queue_id.

        Returns:
            JobQueueStatus with queue details.

        Raises:
            ValueError: If neither job_queue_id nor name is provided.
        """
        # Validation happens in _resolve_to_job_queue_model
        # For status (read operation), project/cloud are optional filters
        raw = self._resolve_to_job_queue_model(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            require_project_and_cloud_with_name=False,
            include_archived=include_archived,
        )
        return _parse_decorated_jq_to_status(raw)

    def update(
        self,
        *,
        job_queue_id: Optional[str] = None,
        job_queue_name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        idle_timeout_s: Optional[int] = None,
    ) -> JobQueueStatus:
        """Update a job queue.

        Args:
            job_queue_id: The ID of the job queue to update.
            job_queue_name: The name of the job queue (alternative to job_queue_id).
            project: The project name (required when using job_queue_name).
            cloud: The cloud name (required when using job_queue_name).
            max_concurrency: New maximum concurrency value.
            idle_timeout_s: New idle timeout in seconds.

        Returns:
            JobQueueStatus with updated queue details.

        Raises:
            ValueError: If neither job_queue_id nor job_queue_name is provided,
                or if job_queue_name is provided without project or cloud.
        """

        if max_concurrency is None and idle_timeout_s is None:
            raise ValueError("No fields to update")

        jq = self._resolve_to_job_queue_model(
            job_queue_id=job_queue_id,
            name=job_queue_name,
            project=project,
            cloud=cloud,
            require_project_and_cloud_with_name=False,
        )

        assert jq.id is not None
        updated_jq = self.client.update_job_queue(
            job_queue_id=jq.id,
            max_concurrency=max_concurrency,
            idle_timeout_s=idle_timeout_s,
        )

        return _parse_decorated_jq_to_status(updated_jq)

    def add_tags(
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Dict[str, str],
    ) -> None:
        if not tags:
            raise ValueError("At least one tag must be provided.")

        if job_queue_id is not None:
            resource_id = job_queue_id
        else:
            if name is None:
                raise ValueError("Either 'job_queue_id' or 'name' must be provided.")
            jq = self._resolve_to_job_queue_model(job_queue_id=None, name=name)
            if jq.id is None:
                raise RuntimeError(f"Job queue with name '{name}' has no ID.")
            resource_id = jq.id

        self.client.upsert_resource_tags(
            ResourceTagResourceType.JOB_QUEUE, resource_id, tags
        )

    def remove_tags(
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        keys: List[str],
    ) -> None:
        if not keys:
            raise ValueError("At least one tag key must be provided.")

        if job_queue_id is not None:
            resource_id = job_queue_id
        else:
            if name is None:
                raise ValueError("Either 'job_queue_id' or 'name' must be provided.")
            jq = self._resolve_to_job_queue_model(job_queue_id=None, name=name)
            if jq.id is None:
                raise RuntimeError(f"Job queue with name '{name}' has no ID.")
            resource_id = jq.id

        self.client.delete_resource_tags(
            ResourceTagResourceType.JOB_QUEUE, resource_id, keys
        )

    def _resolve_to_job_queue_model(
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        require_project_and_cloud_with_name: bool = False,
        include_archived: bool = False,
    ) -> DecoratedJobQueue:
        """Finds the specific Job Queue API model by ID or name.

        Args:
            job_queue_id: The ID of the job queue.
            name: The name of the job queue (alternative to job_queue_id).
            project: The project name to filter by when using name.
            cloud: The cloud name to filter by when using name.
            require_project_and_cloud_with_name: If True, raises ValueError when name
                is provided without project or cloud. Used by archive/terminate to
                prevent ambiguous name resolution.
            include_archived: If True, include archived job queues in the search.
        """
        if job_queue_id is None and name is None:
            raise ValueError("Either 'job_queue_id' or 'name' must be provided.")

        if job_queue_id:
            job_queue = self.client.get_job_queue(job_queue_id)
            if job_queue is None:
                raise ValueError(f"Job Queue with ID '{job_queue_id}' not found.")
            return job_queue
        else:
            if require_project_and_cloud_with_name:
                if project is None:
                    raise ValueError(
                        "'project' is required when using 'name' for this operation."
                    )
                if cloud is None:
                    raise ValueError(
                        "'cloud' is required when using 'name' for this operation."
                    )
            job_queues_response = self.client.list_job_queues(
                name=name,
                project=project,
                cloud=cloud,
                count=1,
                include_archived=include_archived,
            )
            if len(job_queues_response.results) == 0:
                if project and cloud:
                    raise ValueError(
                        f"Job Queue with name '{name}' in project '{project}' "
                        f"and cloud '{cloud}' not found."
                    )
                if project:
                    raise ValueError(
                        f"Job Queue with name '{name}' in project '{project}' not found."
                    )
                raise ValueError(f"Job Queue with name '{name}' not found.")
            return job_queues_response.results[0]

    def list_tags(
        self, *, job_queue_id: Optional[str] = None, name: Optional[str] = None,
    ) -> Dict[str, str]:
        """List tags for a job queue as a key/value mapping."""
        if job_queue_id is not None:
            resource_id = job_queue_id
        else:
            jq = self._resolve_to_job_queue_model(job_queue_id=None, name=name)
            if jq.id is None:
                raise RuntimeError(f"Job queue with name '{name}' has no ID.")
            resource_id = jq.id
        records = self.client.list_resource_tags(
            ResourceTagResourceType.JOB_QUEUE, resource_id
        )
        return {r.key: r.value for r in records if r and r.key is not None}

    def archive(
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
    ) -> str:
        """Archive (seal) a job queue. No new jobs can be submitted after archiving.

        Args:
            job_queue_id: The ID of the job queue to archive.
            name: The name of the job queue (alternative to job_queue_id).
            project: The project name (required when using name).
            cloud: The cloud name (required when using name).

        Returns:
            The ID of the archived job queue.

        Raises:
            ValueError: If neither job_queue_id nor name is provided, or if name
                is provided without project or cloud.
        """
        jq = self._resolve_to_job_queue_model(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            require_project_and_cloud_with_name=True,
        )
        assert jq.id is not None

        self.client.archive_job_queue(jq.id)
        self.logger.info(f"Job queue '{jq.name}' (ID: {jq.id}) has been archived.")
        return jq.id

    def terminate(
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        """Terminate a job queue and all its pending/running jobs.

        Args:
            job_queue_id: The ID of the job queue to terminate.
            name: The name of the job queue (alternative to job_queue_id).
            project: The project name (required when using name).
            cloud: The cloud name (required when using name).
            include_archived: If True, include archived job queues when searching
                by name. Ignored when using job_queue_id.

        Returns:
            The ID of the terminated job queue.

        Raises:
            ValueError: If neither job_queue_id nor name is provided, or if name
                is provided without project or cloud.
        """
        jq = self._resolve_to_job_queue_model(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            require_project_and_cloud_with_name=True,
            include_archived=include_archived,
        )
        assert jq.id is not None

        self.client.terminate_job_queue(jq.id)
        self.logger.info(
            f"Job queue '{jq.name}' (ID: {jq.id}) has been marked for termination."
        )
        return jq.id

    def delete(
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        """Delete a job queue.

        Jobs previously submitted to the queue remain accessible.
        The job queue must have all jobs in terminal state and no running clusters.
        This action permanently removes the job queue and cannot be undone.

        Args:
            job_queue_id: The ID of the job queue to delete.
            name: The name of the job queue (alternative to job_queue_id).
            project: The project name (required when using name).
            cloud: The cloud name (required when using name).
            include_archived: If True, include archived job queues when searching
                by name. Ignored when using job_queue_id.

        Returns:
            The ID of the deleted job queue.

        Raises:
            ValueError: If neither job_queue_id nor name is provided, or if name
                is provided without project or cloud.
            RuntimeError: If the job queue has active jobs or running clusters.
        """
        jq = self._resolve_to_job_queue_model(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            require_project_and_cloud_with_name=True,
            include_archived=include_archived,
        )
        assert jq.id is not None

        self.client.delete_job_queue(jq.id)
        self.logger.info(f"Job queue '{jq.name}' (ID: {jq.id}) has been deleted.")
        return jq.id


def _parse_decorated_jq_to_status(decorated_jq: DecoratedJobQueue) -> JobQueueStatus:
    """Helper to convert API model to SDK model."""

    if decorated_jq.id is None or decorated_jq.current_job_queue_state is None:
        raise ValueError("Job Queue ID or state is missing.")

    return JobQueueStatus(
        id=decorated_jq.id,
        name=decorated_jq.name,
        state=decorated_jq.current_job_queue_state,
        creator_email=decorated_jq.creator_email,
        project_id=decorated_jq.project_id,
        created_at=decorated_jq.created_at,
        max_concurrency=decorated_jq.max_concurrency,
        idle_timeout_s=decorated_jq.idle_timeout_sec,
        creator_id=decorated_jq.creator_id,
        cloud_id=decorated_jq.cloud_id,
        user_provided_id=decorated_jq.user_provided_id,
        execution_mode=decorated_jq.execution_mode,
        total_jobs=decorated_jq.total_jobs,
        active_jobs=decorated_jq.active_jobs,
        successful_jobs=decorated_jq.successful_jobs,
        failed_jobs=decorated_jq.failed_jobs,
    )
