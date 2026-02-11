from typing import Any, Dict, List, Optional, Union

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.job._private.job_sdk import PrivateJobSDK
from anyscale.job.commands import (
    _ADD_TAGS_ARG_DOCSTRINGS,
    _ADD_TAGS_EXAMPLE,
    _ARCHIVE_ARG_DOCSTRINGS,
    _ARCHIVE_EXAMPLE,
    _DELETE_ARG_DOCSTRINGS,
    _DELETE_EXAMPLE,
    _GET_LOGS_ARG_DOCSTRINGS,
    _GET_LOGS_EXAMPLE,
    _LIST_ARG_DOCSTRINGS as _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE as _LIST_EXAMPLE,
    _LIST_TAGS_ARG_DOCSTRINGS,
    _LIST_TAGS_EXAMPLE,
    _REMOVE_TAGS_ARG_DOCSTRINGS,
    _REMOVE_TAGS_EXAMPLE,
    _resolve_id_from_args,
    _STATUS_ARG_DOCSTRINGS,
    _STATUS_EXAMPLE,
    _SUBMIT_ARG_DOCSTRINGS,
    _SUBMIT_EXAMPLE,
    _TERMINATE_ARG_DOCSTRINGS,
    _TERMINATE_EXAMPLE,
    _WAIT_ARG_DOCSTRINGS,
    _WAIT_EXAMPLE,
    add_tags as add_tags,
    archive as archive,
    delete as delete,
    get_logs as get_logs,
    list as list,  # noqa: A004
    list_tags as list_tags,
    remove_tags as remove_tags,
    status as status,
    submit as submit,
    terminate as terminate,
    wait as wait,
)
from anyscale.job.models import JobConfig, JobLogMode, JobState, JobStatus


class JobSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateJobSDK(client=client, logger=logger, timer=timer)

    @sdk_docs(
        doc_py_example=_SUBMIT_EXAMPLE, arg_docstrings=_SUBMIT_ARG_DOCSTRINGS,
    )
    def submit(self, config: JobConfig,) -> str:  # noqa: F811
        """Submit a job.

        Returns the id of the submitted job.
        """
        return self._private_sdk.submit(config=config)

    @sdk_docs(
        doc_py_example=_STATUS_EXAMPLE, arg_docstrings=_STATUS_ARG_DOCSTRINGS,
    )
    def status(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        **_kwargs: Dict[str, Any],
    ) -> JobStatus:
        """Get the status of a job."""
        id = _resolve_id_from_args(id, _kwargs)  # noqa: A001
        return self._private_sdk.status(
            name=name,
            job_id=id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_TERMINATE_EXAMPLE, arg_docstrings=_TERMINATE_ARG_DOCSTRINGS,
    )
    def terminate(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        **_kwargs: Dict[str, Any],
    ) -> str:
        """Terminate a job.

        This command is asynchronous, so it always returns immediately.

        Returns the ID of the terminated job.
        """
        id = _resolve_id_from_args(id, _kwargs)  # noqa: A001
        return self._private_sdk.terminate(
            name=name,
            job_id=id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_ARCHIVE_EXAMPLE, arg_docstrings=_ARCHIVE_ARG_DOCSTRINGS,
    )
    def archive(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        **_kwargs: Dict[str, Any],
    ) -> str:
        """Archive a job.

        This command is asynchronous, so it always returns immediately.

        Returns the ID of the archived job.
        """
        id = _resolve_id_from_args(id, _kwargs)  # noqa: A001
        return self._private_sdk.archive(
            name=name,
            job_id=id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_DELETE_EXAMPLE, arg_docstrings=_DELETE_ARG_DOCSTRINGS,
    )
    def delete(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
        **_kwargs: Dict[str, Any],
    ) -> str:
        """Delete a job and all associated job runs.

        The job must be in a terminal state (SUCCEEDED, FAILED).
        This action permanently removes the job and cannot be undone.

        Returns the ID of the deleted job.
        """
        id = _resolve_id_from_args(id, _kwargs)  # noqa: A001
        return self._private_sdk.delete(
            name=name,
            job_id=id,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_WAIT_EXAMPLE, arg_docstrings=_WAIT_ARG_DOCSTRINGS,
    )
    def wait(  # noqa: F811
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        state: Union[JobState, str] = JobState.SUCCEEDED,
        timeout_s: float = 1800,
        include_archived: bool = False,
        **_kwargs: Dict[str, Any],
    ) -> str:
        """"Wait for a job to enter a specific state."""
        id = _resolve_id_from_args(id, _kwargs)  # noqa: A001
        return self._private_sdk.wait(
            name=name,
            job_id=id,
            cloud=cloud,
            project=project,
            state=state,
            timeout_s=timeout_s,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_GET_LOGS_EXAMPLE, arg_docstrings=_GET_LOGS_ARG_DOCSTRINGS,
    )
    def get_logs(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        run: Optional[str] = None,
        mode: Union[str, JobLogMode] = JobLogMode.TAIL,
        max_lines: Optional[int] = None,
        include_archived: bool = False,
        **_kwargs: Dict[str, Any],
    ) -> str:
        """"Wait for a job to enter a specific state."""
        id = _resolve_id_from_args(id, _kwargs)  # noqa: A001
        return self._private_sdk.get_logs(
            job_id=id,
            name=name,
            cloud=cloud,
            project=project,
            run=run,
            mode=mode,
            max_lines=max_lines,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_ADD_TAGS_EXAMPLE, arg_docstrings=_ADD_TAGS_ARG_DOCSTRINGS,
    )
    def add_tags(  # noqa: F811
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        tags: Dict[str, str],
        include_archived: bool = False,
    ) -> None:
        """Upsert (add/update) tag key/value pairs for a job."""
        return self._private_sdk.add_tags(
            job_id=job_id,
            name=name,
            cloud=cloud,
            project=project,
            tags=tags,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_REMOVE_TAGS_EXAMPLE, arg_docstrings=_REMOVE_TAGS_ARG_DOCSTRINGS,
    )
    def remove_tags(  # noqa: F811
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        keys: list,
        include_archived: bool = False,
    ) -> None:
        """Remove tags by key from a job."""
        return self._private_sdk.remove_tags(
            job_id=job_id,
            name=name,
            cloud=cloud,
            project=project,
            keys=keys,
            include_archived=include_archived,
        )

    @sdk_docs(
        doc_py_example=_LIST_TAGS_EXAMPLE, arg_docstrings=_LIST_TAGS_ARG_DOCSTRINGS
    )
    def list_tags(  # noqa: F811
        self,
        *,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        include_archived: bool = False,
    ) -> Dict[str, str]:
        """List tags for a job."""
        return self._private_sdk.list_tags(
            job_id=job_id,
            name=name,
            cloud=cloud,
            project=project,
            include_archived=include_archived,
        )

    @sdk_docs(doc_py_example=_LIST_EXAMPLE, arg_docstrings=_LIST_ARG_DOCSTRINGS)
    def list(  # noqa: PLR0913
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        project: Optional[str] = None,
        project_id: Optional[str] = None,
        cloud: Optional[str] = None,
        include_all_users: bool = False,
        include_archived: bool = False,
        state_filter: Optional[List[Union[JobState, str]]] = None,
        tags_filter: Optional[Dict[str, List[str]]] = None,
        page_size: Optional[int] = None,
        max_items: Optional[int] = None,
        sort_field: Optional[str] = None,
        sort_order: Optional[str] = None,
        **_kwargs: Dict[str, Any],
    ) -> ResultIterator[JobStatus]:
        """List jobs with filtering and pagination.

        Returns a ResultIterator that lazily fetches pages of jobs.

        Args:
            name: Filter by job name.
            id: Fetch a specific job by ID.
            project: Filter by project name.
            project_id: [DEPRECATED] Filter by project ID. Use 'project' instead.
            cloud: Filter by cloud name.
            include_all_users: Include jobs from all users.
            include_archived: Include archived jobs.
            state_filter: Filter by job states (list of JobState or str).
            tags_filter: Filter by tags (dict of key to list of values).
            page_size: Number of items per page. Defaults to server default.
            max_items: Maximum total items to return.
            sort_field: Field to sort by (CREATED_AT, NAME, STATUS, etc.).
            sort_order: Sort order (ASC or DESC).

        Returns:
            ResultIterator of JobStatus objects.
        """
        id = _resolve_id_from_args(id, _kwargs)  # noqa: A001
        return self._private_sdk.list(
            name=name,
            job_id=id,
            project=project,
            project_id=project_id,
            cloud=cloud,
            include_all_users=include_all_users,
            include_archived=include_archived,
            state_filter=state_filter,
            tags_filter=tags_filter,
            page_size=page_size,
            max_items=max_items,
            sort_field=sort_field,
            sort_order=sort_order,
        )
