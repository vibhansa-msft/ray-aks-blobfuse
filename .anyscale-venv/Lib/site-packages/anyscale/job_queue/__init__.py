from typing import Dict, List, Optional

from anyscale._private.anyscale_client import AnyscaleClient
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_command as sdk_command, sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.job_queue._private.job_queue_sdk import PrivateJobQueueSDK
from anyscale.job_queue.commands import (
    _ARCHIVE_ARG_DOCSTRINGS,
    _ARCHIVE_EXAMPLE,
    _DELETE_ARG_DOCSTRINGS,
    _DELETE_EXAMPLE,
    _JOB_QUEUE_SDK_SINGLETON_KEY as _JOB_QUEUE_SDK_SINGLETON_KEY,
    _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE,
    _STATUS_ARG_DOCSTRINGS,
    _STATUS_EXAMPLE,
    _TAGS_ADD_ARG_DOCSTRINGS,
    _TAGS_ADD_EXAMPLE,
    _TAGS_LIST_ARG_DOCSTRINGS,
    _TAGS_LIST_EXAMPLE,
    _TAGS_REMOVE_ARG_DOCSTRINGS,
    _TAGS_REMOVE_EXAMPLE,
    _TERMINATE_ARG_DOCSTRINGS,
    _TERMINATE_EXAMPLE,
    _UPDATE_ARG_DOCSTRINGS,
    _UPDATE_EXAMPLE,
    add_tags as add_tags,
    archive as archive,
    delete as delete,
    list as list,  # noqa: A004 - claude_comment("claude-opus-4-5", "SDK public API re-export")
    list_tags as list_tags,
    remove_tags as remove_tags,
    status as status,
    terminate as terminate,
    update as update,
)
from anyscale.job_queue.models import (
    JobQueueSortDirective,
    JobQueueSortField as JobQueueSortField,
    JobQueueState as JobQueueState,
    JobQueueStatus,
    SessionState,
)


class JobQueueSDK:
    """Public SDK for interacting with Anyscale Job Queues."""

    def __init__(
        self,
        *,
        client: Optional[AnyscaleClient] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateJobQueueSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS)
    def list(  # noqa: F811, PLR0913
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
        """List job queues or fetch a single job queue by ID."""
        return self._private_sdk.list(
            job_queue_id=job_queue_id,
            name=name,
            creator_id=creator_id,
            cloud=cloud,
            project=project,
            cluster_status=cluster_status,
            tags_filter=tags_filter,
            page_size=page_size,
            max_items=max_items,
            sorting_directives=sorting_directives,
            include_archived=include_archived,
        )

    @sdk_docs(doc_py_example=_STATUS_EXAMPLE, arg_docstrings=_STATUS_ARG_DOCSTRINGS)
    def status(  # noqa: F811
        self,
        job_queue_id: Optional[str] = None,
        *,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> JobQueueStatus:
        """Get the status and details for a specific job queue."""
        return self._private_sdk.status(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            include_archived=include_archived,
        )

    @sdk_docs(doc_py_example=_UPDATE_EXAMPLE, arg_docstrings=_UPDATE_ARG_DOCSTRINGS)
    def update(  # noqa: F811
        self,
        *,
        job_queue_id: Optional[str] = None,
        job_queue_name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        idle_timeout_s: Optional[int] = None,
    ) -> JobQueueStatus:
        """Update a job queue."""
        return self._private_sdk.update(
            job_queue_id=job_queue_id,
            job_queue_name=job_queue_name,
            project=project,
            cloud=cloud,
            max_concurrency=max_concurrency,
            idle_timeout_s=idle_timeout_s,
        )

    @sdk_docs(doc_py_example=_TAGS_ADD_EXAMPLE, arg_docstrings=_TAGS_ADD_ARG_DOCSTRINGS)
    def add_tags(  # noqa: F811
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Dict[str, str],
    ) -> None:
        """Upsert (add/update) tag key/value pairs for a job queue."""
        return self._private_sdk.add_tags(
            job_queue_id=job_queue_id, name=name, tags=tags
        )

    @sdk_docs(
        doc_py_example=_TAGS_REMOVE_EXAMPLE, arg_docstrings=_TAGS_REMOVE_ARG_DOCSTRINGS
    )
    def remove_tags(  # noqa: F811
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        keys: List[str],
    ) -> None:
        """Remove tags by key from a job queue."""
        return self._private_sdk.remove_tags(
            job_queue_id=job_queue_id, name=name, keys=keys
        )

    @sdk_docs(
        doc_py_example=_TAGS_LIST_EXAMPLE, arg_docstrings=_TAGS_LIST_ARG_DOCSTRINGS
    )
    def list_tags(  # noqa: F811
        self, *, job_queue_id: Optional[str] = None, name: Optional[str] = None,
    ) -> Dict[str, str]:
        """List tags for a job queue."""
        return self._private_sdk.list_tags(job_queue_id=job_queue_id, name=name)

    @sdk_docs(doc_py_example=_ARCHIVE_EXAMPLE, arg_docstrings=_ARCHIVE_ARG_DOCSTRINGS)
    def archive(  # noqa: F811
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
    ) -> str:
        """Archive (seal) a job queue. No new jobs can be submitted."""
        return self._private_sdk.archive(
            job_queue_id=job_queue_id, name=name, project=project, cloud=cloud
        )

    @sdk_docs(
        doc_py_example=_TERMINATE_EXAMPLE, arg_docstrings=_TERMINATE_ARG_DOCSTRINGS
    )
    def terminate(  # noqa: F811
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        """Terminate a job queue and all its pending/running jobs."""
        return self._private_sdk.terminate(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            include_archived=include_archived,
        )

    @sdk_docs(doc_py_example=_DELETE_EXAMPLE, arg_docstrings=_DELETE_ARG_DOCSTRINGS)
    def delete(  # noqa: F811
        self,
        *,
        job_queue_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        include_archived: bool = False,
    ) -> str:
        """Delete a job queue. Jobs previously submitted remain accessible.

        The job queue must have all jobs in terminal state and no running clusters.
        This action cannot be undone.
        """
        return self._private_sdk.delete(
            job_queue_id=job_queue_id,
            name=name,
            project=project,
            cloud=cloud,
            include_archived=include_archived,
        )
