from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Generic, List, TypeVar

import humanize
import tabulate

from anyscale.client.openapi_client.models.decorated_list_service_api_model import (
    DecoratedListServiceAPIModel,
)
from anyscale.client.openapi_client.models.decorated_schedule import DecoratedSchedule


TR = TypeVar("TR")


class AbstractTable(ABC, Generic[TR]):
    """An Abstract table class """

    entity_name = "object"

    def __init__(self, rows: List[TR]):
        self.rows = rows

    @abstractmethod
    def render_row(self, row: TR) -> Dict[str, str]:
        """Render one row of the table

        Args:
            row (TR): A row object.

        Returns:
            Dict[str, str]: A map from column headers to row values. Ordering is preserved. The order of the headers
                in the dict is the order that will be shown in the table. This function should always return a dict with the same keys,
                in the same order.
        """
        ...

    def __str__(self):
        if len(self.rows) == 0:
            return f"No {self.entity_name}s found."
        rendered_rows = [self.render_row(row) for row in self.rows]
        headers = list(rendered_rows[0].keys())
        rows = [row.values() for row in rendered_rows]
        return tabulate.tabulate(rows, headers=headers, tablefmt="psql")


class SchedulesTable(AbstractTable[DecoratedSchedule]):
    entity_name = "Schedule"

    def render_row(self, decorated_schedule: DecoratedSchedule):
        last_exec_list = decorated_schedule.last_executions
        most_recent_exec_id = last_exec_list[0].id if len(last_exec_list) > 0 else ""

        return {
            "ID": decorated_schedule.id,
            "NAME": decorated_schedule.name,
            "DESCRIPTION": decorated_schedule.description,
            "PROJECT": decorated_schedule.project.name,
            "CRON": decorated_schedule.schedule.cron_expression,
            "NEXT TRIGGER": humanize.naturaltime(
                datetime.now(timezone.utc) - decorated_schedule.next_trigger_at
            )
            if decorated_schedule.next_trigger_at
            else "Schedule is paused",
            "TIMEZONE": decorated_schedule.schedule.timezone,
            "CREATOR": decorated_schedule.creator.email,
            "LATEST EXECUTION ID": most_recent_exec_id,
        }


class ServicesTable(AbstractTable[DecoratedListServiceAPIModel]):
    entity_name = "Service"

    def render_row(self, service: DecoratedListServiceAPIModel):
        # TODO(shawnp): add creator username once that's exposed in the API.
        return {
            "NAME": service.name,
            "SERVICE_ID": service.id,
            "PROJECT_ID": service.project_id,
            "CURRENT_STATE": service.current_state,
        }
