from typing import Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.models.model_base import ResultIterator
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.cli_logger import BlockLogger
from anyscale.schedule._private.schedule_sdk import PrivateScheduleSDK
from anyscale.schedule.commands import (
    _APPLY_ARG_DOCSTRINGS,
    _APPLY_EXAMPLE,
    _DELETE_ARG_DOCSTRINGS,
    _DELETE_EXAMPLE,
    _LIST_ARG_DOCSTRINGS as _LIST_ARG_DOCSTRINGS,
    _LIST_EXAMPLE as _LIST_EXAMPLE,
    _SET_STATE_ARG_DOCSTRINGS,
    _SET_STATE_EXAMPLE,
    _STATUS_ARG_DOCSTRINGS,
    _STATUS_EXAMPLE,
    _TRIGGER_ARG_DOCSTRINGS,
    _TRIGGER_EXAMPLE,
    _URL_ARG_DOCSTRINGS,
    _URL_EXAMPLE,
    apply as apply,
    delete as delete,
    list as list,  # noqa: A004
    set_state as set_state,
    status as status,
    trigger as trigger,
    url as url,
)
from anyscale.schedule.models import ScheduleConfig, ScheduleState, ScheduleStatus


class ScheduleSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateScheduleSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_APPLY_EXAMPLE, arg_docstrings=_APPLY_ARG_DOCSTRINGS,
    )
    def apply(self, config: ScheduleConfig,) -> str:  # noqa: F811
        """Apply or update a schedule.

        Returns the id of the schedule.
        """
        return self._private_sdk.apply(config=config)

    @sdk_docs(
        doc_py_example=_SET_STATE_EXAMPLE, arg_docstrings=_SET_STATE_ARG_DOCSTRINGS,
    )
    def set_state(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
        state: ScheduleState,
    ) -> str:  # noqa: F811
        """Set the state of a schedule.

        Returns the id of the schedule.
        """
        return self._private_sdk.set_state(
            id=id, name=name, cloud=cloud, project=project, state=state,
        )

    @sdk_docs(doc_py_example=_STATUS_EXAMPLE, arg_docstrings=_STATUS_ARG_DOCSTRINGS)
    def status(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> ScheduleStatus:
        """Return the status of the schedule.
        """
        return self._private_sdk.status(id=id, name=name, cloud=cloud, project=project)

    @sdk_docs(doc_py_example=_TRIGGER_EXAMPLE, arg_docstrings=_TRIGGER_ARG_DOCSTRINGS)
    def trigger(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Trigger the execution of the schedule.
        """
        return self._private_sdk.trigger(id=id, name=name, cloud=cloud, project=project)

    @sdk_docs(doc_py_example=_URL_EXAMPLE, arg_docstrings=_URL_ARG_DOCSTRINGS)
    def url(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Get the web UI URL for a schedule."""
        return self._private_sdk.url(id=id, name=name, cloud=cloud, project=project)

    def list(  # noqa: F811 PLR0913
        self,
        *,
        name: Optional[str] = None,
        schedule_id: Optional[str] = None,
        project: Optional[str] = None,
        cloud: Optional[str] = None,
        creator_id: Optional[str] = None,
        include_all_users: bool = False,
        page_size: int = 10,
        max_items: Optional[int] = None,
    ) -> ResultIterator[ScheduleStatus]:
        """List schedules with filtering and pagination.

        Returns a ResultIterator that lazily fetches pages of schedules.

        Args:
            name: Filter by schedule name.
            schedule_id: Fetch a specific schedule by ID.
            project: Filter by project name.
            cloud: Filter by cloud name.
            creator_id: Filter by creator ID.
            include_all_users: Include schedules from all users.
            page_size: Number of items per page.
            max_items: Maximum total items to return.

        Returns:
            ResultIterator of ScheduleStatus objects.
        """
        return self._private_sdk.list(
            name=name,
            schedule_id=schedule_id,
            project=project,
            cloud=cloud,
            creator_id=creator_id,
            include_all_users=include_all_users,
            page_size=page_size,
            max_items=max_items,
        )

    @sdk_docs(doc_py_example=_DELETE_EXAMPLE, arg_docstrings=_DELETE_ARG_DOCSTRINGS)
    def delete(  # noqa: F811
        self,
        *,
        id: Optional[str] = None,  # noqa: A002
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        project: Optional[str] = None,
    ) -> str:
        """Delete a schedule.

        If the schedule is active, it will be automatically paused before deletion.
        The schedule must have no active triggered jobs.
        Returns the ID of the deleted schedule.
        """
        return self._private_sdk.delete(id=id, name=name, cloud=cloud, project=project)
