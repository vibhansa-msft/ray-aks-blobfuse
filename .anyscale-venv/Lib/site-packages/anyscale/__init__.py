import inspect
import logging
import os
from sys import path
from typing import Any, Dict, List, Optional
import warnings

import click

from anyscale import version


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("ANYSCALE_LOGLEVEL", "WARN"))

anyscale_dir = os.path.dirname(os.path.abspath(__file__))
path.append(os.path.join(anyscale_dir, "client"))
path.append(os.path.join(anyscale_dir, "sdk"))

import anyscale as anyscale
from anyscale import (
    aggregated_instance_usage as aggregated_instance_usage,
    cloud as cloud,
    compute_config as compute_config,
    image as image,
    integrations as integrations,
    job as job,
    job_queue as job_queue,
    organization_invitation as organization_invitation,
    policy as policy,
    project as project,
    resource_quota as resource_quota,
    schedule as schedule,
    service as service,
    service_account as service_account,
    user as user,
    user_group as user_group,
)
from anyscale._private.anyscale_client import AnyscaleClient, AnyscaleClientInterface
from anyscale._private.sdk.base_sdk import Timer
from anyscale.aggregated_instance_usage import AggregatedInstanceUsageSDK
from anyscale.authenticate import AuthenticationBlock
from anyscale.cli_logger import BlockLogger
from anyscale.cloud import CloudSDK
from anyscale.cluster import (
    get_job_submission_client_cluster_info as get_job_submission_client_cluster_info,
)
from anyscale.cluster_compute import (
    get_cluster_compute_from_name as get_cluster_compute_from_name,
)
from anyscale.compute_config import ComputeConfigSDK
from anyscale.connect import ClientBuilder
from anyscale.image import ImageSDK
from anyscale.job import JobSDK
from anyscale.job_queue import JobQueueSDK
from anyscale.organization_invitation import OrganizationInvitationSDK
from anyscale.policy import PolicySDK
from anyscale.project import ProjectSDK
from anyscale.resource_quota import ResourceQuotaSDK
from anyscale.schedule import ScheduleSDK
from anyscale.service import ServiceSDK
from anyscale.service_account import ServiceAccountSDK
from anyscale.user import UserSDK
from anyscale.user_group import UserGroupSDK
from anyscale.workspace import WorkspaceSDK


# Note: indentation here matches that of connect.py::ClientBuilder.
BUILDER_HELP_FOOTER = """
        See ``anyscale.ClientBuilder`` for full documentation of
        this experimental feature."""

# Auto-add all Anyscale connect builder functions to the top-level.
for attr, _ in inspect.getmembers(ClientBuilder, inspect.isfunction):
    if attr.startswith("_"):
        continue

    # This is exposed in anyscale/cloud/__init__.py since anyscale.cloud is used as the SDK module too.
    if attr == "cloud":
        continue

    def _new_builder(attr: str) -> Any:
        target = getattr(ClientBuilder, attr)

        def new_session_builder(*a: List[Any], **kw: Dict[str, Any]) -> Any:
            builder = ClientBuilder()
            return target(builder, *a, **kw)

        new_session_builder.__name__ = attr
        new_session_builder.__doc__ = target.__doc__ + BUILDER_HELP_FOOTER
        new_session_builder.__signature__ = inspect.signature(target)  # type: ignore

        return new_session_builder

    globals()[attr] = _new_builder(attr)

__version__ = version.__version__

ANYSCALE_ENV = os.environ.copy()

# Remove this code once AnyscaleSDK is removed
# Keep the old import for backwards compatibility but warn
def __getattr__(name):
    if name == "AnyscaleSDK":
        warnings.warn(
            "Anyscale has deprecated the legacy AnyscaleSDK class. "
            "Anyscale will remove this and its methods on February 28, 2026.\n\n"
            "The Anyscale SDK remains fully supported but uses a new pattern. "
            "Migrate to the maintained SDK by using `import anyscale` and calling "
            "`anyscale.<module>.<function>()`.\n\n"
            "See https://docs.anyscale.com/reference#sdk",
            DeprecationWarning,
            stacklevel=2,
        )
        # Lazy import to avoid binding at module import time, ensuring the warning is shown on import.
        from anyscale.sdk.anyscale_client.sdk import (  # noqa: PLC0415 - deprecated import; will be removed in a future version
            AnyscaleSDK,
        )

        return AnyscaleSDK
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Anyscale:
    def __init__(
        self,
        *,
        auth_token: Optional[str] = None,
        _host: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        auth_block = AuthenticationBlock(
            cli_token=auth_token, host=_host, raise_structured_exception=True,
        )

        _validate_headers(headers)

        if headers:
            for k, v in headers.items():
                auth_block.api_client.api_client.set_default_header(k, v)
                auth_block.anyscale_api_client.api_client.set_default_header(k, v)

        self._anyscale_client = AnyscaleClient(
            api_clients=(auth_block.anyscale_api_client, auth_block.api_client),
            host=_host,
        )
        self._aggregated_instance_usage_sdk = AggregatedInstanceUsageSDK(
            client=self._anyscale_client
        )
        self._job_sdk = JobSDK(client=self._anyscale_client)
        self._job_queue_sdk = JobQueueSDK(client=self._anyscale_client)
        self._service_sdk = ServiceSDK(client=self._anyscale_client)
        self._compute_config_sdk = ComputeConfigSDK(client=self._anyscale_client)
        self._cloud_sdk = CloudSDK(client=self._anyscale_client)
        self._schedule_sdk = ScheduleSDK(client=self._anyscale_client)
        self._image_sdk = ImageSDK(client=self._anyscale_client)
        self._organization_invitation_sdk = OrganizationInvitationSDK(
            client=self._anyscale_client
        )
        self._policy_sdk = PolicySDK(client=self._anyscale_client)
        self._project_sdk = ProjectSDK(client=self._anyscale_client)
        self._user_group_sdk = UserGroupSDK(client=self._anyscale_client)
        self._resource_quota_sdk = ResourceQuotaSDK(client=self._anyscale_client)
        self._service_account_sdk = ServiceAccountSDK(client=self._anyscale_client)
        self._user_sdk = UserSDK(client=self._anyscale_client)
        self._workspace_sdk = WorkspaceSDK(client=self._anyscale_client)

    @classmethod
    def _init_private(
        cls, *, client: AnyscaleClientInterface, logger: BlockLogger, timer: Timer,
    ):
        # Private constructor used to inject fakes for testing.
        obj = cls.__new__(cls)
        super(Anyscale, obj).__init__()
        obj._anyscale_client = client  # noqa: SLF001
        obj._aggregated_instance_usage_sdk = AggregatedInstanceUsageSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._job_sdk = JobSDK(client=client, logger=logger, timer=timer)  # noqa: SLF001
        obj._job_queue_sdk = JobQueueSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._service_sdk = ServiceSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._compute_config_sdk = ComputeConfigSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._cloud_sdk = CloudSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._schedule_sdk = ScheduleSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer,
        )
        obj._image_sdk = ImageSDK(client=client, logger=logger)  # noqa: SLF001
        obj._organization_invitation_sdk = OrganizationInvitationSDK(  # noqa: SLF001
            client=client, logger=logger
        )
        obj._policy_sdk = PolicySDK(client=client, logger=logger)  # noqa: SLF001
        obj._project_sdk = ProjectSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._user_group_sdk = UserGroupSDK(client=client, logger=logger)  # noqa: SLF001
        obj._service_account_sdk = ServiceAccountSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._user_sdk = UserSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._resource_quota_sdk = ResourceQuotaSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer
        )
        obj._workspace_sdk = WorkspaceSDK(  # noqa: SLF001
            client=client, logger=logger, timer=timer,
        )
        return obj

    @property
    def aggregated_instance_usage(self) -> AggregatedInstanceUsageSDK:  # noqa: F811
        return self._aggregated_instance_usage_sdk

    @property
    def job(self) -> JobSDK:  # noqa: F811
        return self._job_sdk

    @property
    def job_queue(self) -> JobQueueSDK:  # noqa: F811
        return self._job_queue_sdk

    @property
    def service(self) -> ServiceSDK:  # noqa: F811
        return self._service_sdk

    @property
    def compute_config(self) -> ComputeConfigSDK:  # noqa: F811
        return self._compute_config_sdk

    @property
    def cloud(self) -> CloudSDK:  # noqa: F811
        return self._cloud_sdk

    @property
    def schedule(self) -> ScheduleSDK:  # noqa: F811
        return self._schedule_sdk

    @property
    def image(self) -> ImageSDK:  # noqa: F811
        return self._image_sdk

    @property
    def organization_invitation(self) -> OrganizationInvitationSDK:  # noqa: F811
        return self._organization_invitation_sdk

    @property
    def policy(self) -> PolicySDK:  # noqa: F811
        return self._policy_sdk

    @property
    def project(self) -> ProjectSDK:  # noqa: F811
        return self._project_sdk

    @property
    def user_group(self) -> UserGroupSDK:  # noqa: F811
        return self._user_group_sdk

    @property
    def resource_quota(self) -> ResourceQuotaSDK:  # noqa: F811
        return self._resource_quota_sdk

    @property
    def service_account(self) -> ServiceAccountSDK:  # noqa: F811
        return self._service_account_sdk

    @property
    def user(self) -> UserSDK:  # noqa: F811
        return self._user_sdk

    @property
    def workspace(self) -> WorkspaceSDK:  # noqa: F811
        return self._workspace_sdk


def _validate_headers(headers: Optional[Dict[str, str]]):
    if not headers:
        return

    for k, v in headers.items():
        if isinstance(k, str) is False:
            raise click.ClickException(f"The header {k} must be a string.")
        if isinstance(v, str) is False:
            raise click.ClickException(f"The value {v} to header {k} must be a string.")
