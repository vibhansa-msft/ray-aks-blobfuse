from typing import Optional

from anyscale._private.sdk import sdk_command
from anyscale.aggregated_instance_usage._private.aggregated_instance_usage_sdk import (
    PrivateAggregatedInstanceUsageSDK,
)
from anyscale.aggregated_instance_usage.models import DownloadCSVFilters


_AGGREGATED_INSTANCE_USAGE_SDK_SINGLETON_KEY = "aggregated_instance_usage_sdk"

_DOWNLOAD_CSV_EXAMPLE = """
import anyscale
from anyscale.aggregated_instance_usage.models import DownloadCSVFilters

anyscale.aggregated_instance_usage.download_csv(
    DownloadCSVFilters(
        start_date="2024-10-01",
        end_date="2024-10-02",
        cloud="cloud_name",
        project="project_name",
        directory="/directory",
        hide_progress_bar=False,
    ),
)
"""

_DOWNLOAD_ARG_DOCSTRINGS = {"filters": "The filter of the instance usage to download."}


@sdk_command(
    _AGGREGATED_INSTANCE_USAGE_SDK_SINGLETON_KEY,
    PrivateAggregatedInstanceUsageSDK,
    doc_py_example=_DOWNLOAD_CSV_EXAMPLE,
    arg_docstrings=_DOWNLOAD_ARG_DOCSTRINGS,
)
def download_csv(
    filters: DownloadCSVFilters,
    *,
    _private_sdk: Optional[PrivateAggregatedInstanceUsageSDK] = None
) -> str:
    """Download an aggregated instance usage report as a zipped CSV to the provided directory.
    """
    return _private_sdk.download_csv(filters)  # type: ignore
