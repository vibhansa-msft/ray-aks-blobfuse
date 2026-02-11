from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from anyscale._private.models import ModelBase


@dataclass(frozen=True)
class DownloadCSVFilters(ModelBase):
    """Filters to use when downloading usage CSVs.
    """

    __doc_py_example__ = """\
import anyscale
from anyscale.aggregated_instance_usage.models import DownloadCSVFilters

download_csv_filters = DownloadCSVFilters(
    # Start date (UTC inclusive) for the usage CSV.
    start_date="2024-10-01",
    # End date (UTC inclusive) for the usage CSV.
    end_date="2024-10-31",
    # Optional cloud name to filter by.
    cloud="cloud_name",
    # Optional project name to filter by.
    project="project_name",
    # Optional directory to save the CSV to.
    directory="/directory",
    # Optional hide progress bar.
    hide_progress_bar=False,
)
"""

    def _validate_date(self, date_string: str):
        if not isinstance(date_string, str):
            raise TypeError("date must be a string.")
        try:
            date.fromisoformat(date_string)
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYY-MM-DD")

    start_date: str = field(
        metadata={"docstring": "Start date (UTC inclusive) for the usage CSV."}
    )

    def _validate_start_date(self, start_date: str):
        self._validate_date(start_date)

    end_date: str = field(
        metadata={"docstring": "End date (UTC inclusive) for the usage CSV."}
    )

    def _validate_end_date(self, end_date: str):
        self._validate_date(end_date)

    cloud: Optional[str] = field(
        default=None, metadata={"docstring": "Optional cloud name to filter by."},
    )

    def _validate_cloud(self, cloud: Optional[str]):
        if cloud is not None and not isinstance(cloud, str):
            raise TypeError("cloud must be a string.")

    project: Optional[str] = field(
        default=None, metadata={"docstring": "Optional project name to filter by."},
    )

    def _validate_project(self, project: Optional[str]):
        if project is not None and not isinstance(project, str):
            raise TypeError("project must be a string.")

    directory: Optional[str] = field(
        default=None, metadata={"docstring": "Optional directory to save the CSV to."},
    )

    def _validate_directory(self, directory: Optional[str]):
        if directory is not None and not isinstance(directory, str):
            raise TypeError("directory must be a string.")

    hide_progress_bar: Optional[bool] = field(
        default=False, metadata={"docstring": "Optional hide progress bar."},
    )

    def _validate_hide_progress_bar(self, hide_progress_bar: Optional[bool]):
        if hide_progress_bar is not None and not isinstance(hide_progress_bar, bool):
            raise TypeError("hide_progress_bar must be a boolean.")
