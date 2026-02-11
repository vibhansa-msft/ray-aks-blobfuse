from typing import Optional

from anyscale._private.anyscale_client import (
    AnyscaleClient,
    AnyscaleClientInterface,
)
from anyscale._private.sdk.timer import RealTimer, Timer
from anyscale.cli_logger import BlockLogger


class BaseSDK:
    """Shared parent class for all SDKs."""

    def __init__(
        self,
        *,
        logger: Optional[BlockLogger] = None,
        client: Optional[AnyscaleClientInterface] = None,
        timer: Optional[Timer] = None,
    ):
        self._logger = logger or BlockLogger()
        self._client = client or AnyscaleClient()
        self._timer = timer or RealTimer()

    @property
    def logger(self) -> BlockLogger:
        return self._logger

    @property
    def client(self) -> AnyscaleClientInterface:
        return self._client

    @property
    def timer(self) -> Timer:
        return self._timer
