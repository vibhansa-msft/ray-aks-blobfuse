from typing import Optional

from anyscale._private.anyscale_client import AnyscaleClientInterface
from anyscale._private.sdk import sdk_docs
from anyscale._private.sdk.base_sdk import Timer
from anyscale.aggregated_instance_usage._private.aggregated_instance_usage_sdk import (
    PrivateAggregatedInstanceUsageSDK,
)
from anyscale.aggregated_instance_usage.commands import (
    _DOWNLOAD_ARG_DOCSTRINGS,
    _DOWNLOAD_CSV_EXAMPLE,
    download_csv as download_csv,
)
from anyscale.aggregated_instance_usage.models import DownloadCSVFilters
from anyscale.cli_logger import BlockLogger


class AggregatedInstanceUsageSDK:
    def __init__(
        self,
        *,
        client: Optional[AnyscaleClientInterface] = None,
        logger: Optional[BlockLogger] = None,
        timer: Optional[Timer] = None,
    ):
        self._private_sdk = PrivateAggregatedInstanceUsageSDK(
            client=client, logger=logger, timer=timer
        )

    @sdk_docs(
        doc_py_example=_DOWNLOAD_CSV_EXAMPLE, arg_docstrings=_DOWNLOAD_ARG_DOCSTRINGS,
    )
    def download_csv(self, filters: DownloadCSVFilters,) -> str:  # noqa: F811
        """Download an aggregated instance usage report as a zipped CSV to the provided directory.
        """
        return self._private_sdk.download_csv(filters=filters)
