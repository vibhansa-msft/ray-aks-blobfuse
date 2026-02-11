# pylint:disable=private-import
import builtins
from contextlib import contextmanager
import copy
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from urllib.parse import urlparse

import click

from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client import ComputeTemplate, DecoratedCloudResource
from anyscale.client.openapi_client.api.default_api import DefaultApi
from anyscale.client.openapi_client.models.cloud_providers import CloudProviders
from anyscale.client.openapi_client.models.user_info import UserInfo
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as SDKDefaultApi
from anyscale.util import is_anyscale_workspace
from anyscale.utils.ray_utils import zip_directory  # type: ignore
from anyscale.utils.workload_types import Workload


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# TODO(austin): refactor to read requirements.txt and .skip_packages_tracking from s3 or gcs directly.
# Default cluster storage directory.
CLUSTER_STORAGE_DIR = os.environ.get(
    "ANYSCALE_CLUSTER_STORAGE_DIR", "/mnt/cluster_storage"
)

# Location of the directory containing workspace configurations.
WORKSPACE_CONF_DIR = os.environ.get(
    "ANYSCALE_WORKSPACE_CONF_DIR", os.path.join(CLUSTER_STORAGE_DIR, ".anyscale")
)

# Location of the workspace-managed requirements.txt file that is automatically populated.
# This is not always guaranteed to exist within a workspace.
WORKSPACE_REQUIREMENTS_FILE_PATH = os.path.join(WORKSPACE_CONF_DIR, "requirements.txt")

# Feature flags for pip dependency tracking.
SKIP_PACKAGES_TRACKING_PATH = os.path.join(
    WORKSPACE_CONF_DIR, ".skip_packages_tracking"
)


def is_workspace_dependency_tracking_disabled() -> bool:
    """Returns True if the workspace dependency tracking is disabled.

    Enabled iff:
        - ANYSCALE_WORKSPACE_DYNAMIC_DEPENDENCY_TRACKING env var is set to 1.
        - ANYSCALE_SKIP_PYTHON_DEPENDENCY_TRACKING env var is not set to 1.
        - No file exists at SKIP_PACKAGES_TRACKING_PATH.
    """
    # NOTE(edoakes): The environment variable is evaluated here instead of in the global scope
    # so it can easily be overwritten for testing.
    enabled = (
        os.environ.get("ANYSCALE_WORKSPACE_DYNAMIC_DEPENDENCY_TRACKING", "0") == "1"
    )
    return (
        not enabled
        or os.environ.get("ANYSCALE_SKIP_PYTHON_DEPENDENCY_TRACKING", "0") == "1"
        or os.path.exists(SKIP_PACKAGES_TRACKING_PATH)
    )


def _upload_file_to_google_cloud_storage(file: str, bucket: str, object_name: str):
    try:
        from google.cloud import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP storage dependency")
            storage,
        )

    except Exception:  # noqa: BLE001
        raise click.ClickException(
            "Could not upload file to Google Storage. Could not import the Google Storage Python API via `from google.cloud import storage`.  Please check your installation or try running `pip install --upgrade google-cloud-storage`."
        )
    try:
        storage_client = storage.Client()
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(object_name)
        blob.upload_from_filename(file)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(
            f"Failed to upload the working directory to Google Cloud Storage. Error {e!r}"
            "Please validate you have exported cloud credentials with the correct write permissions and the intended bucket exists in your Cloud Storage account. "
            "If you do not desire to upload your working directory, please set your working directory to a public remote URI or remove the runtime_environment from your job or service yaml."
        ) from e


def _upload_file_to_s3(file: str, bucket: str, object_key: str):
    try:
        import boto3  # noqa: PLC0415 - codex_reason("gpt5.2", "optional AWS SDK dependency")
        import botocore.config  # noqa: PLC0415 - codex_reason("gpt5.2", "optional AWS SDK dependency")
    except Exception:  # noqa: BLE001
        raise click.ClickException(
            "Could not upload file to S3: Could not import the Amazon S3 Python API via `import boto3`.  Please check your installation or try running `pip install boto3`."
        )
    try:
        s3_client = boto3.client(
            "s3", config=botocore.config.Config(signature_version="s3v4")
        )
        s3_client.upload_file(file, bucket, object_key)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(
            f"Failed to upload the working directory to S3. Error {e!r}"
            "Please validate you have exported cloud credentials with the correct write permissions and the intended bucket exists in your S3 account. "
            "If you do not desire to upload your working directory, please set your working directory to a public remote URI or remove the runtime_environment from your job or service yaml."
        ) from e


def _upload_file_to_azure_storage(file: str, upload_path: str, object_name: str) -> str:
    """Upload a file to Azure Blob Storage using smart_open (same library Ray uses for downloads).

    Args:
        file: Local file path to upload
        upload_path: Azure storage path (abfss://container@account.dfs.core.windows.net/path)
        object_name: Name of the object to upload

    Returns:
        str: The final uploaded filepath (azure://container/path/filename)

    Raises:
        click.ClickException: If upload fails or required packages are missing
    """
    try:
        from azure.identity import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional Azure identity dependency")
            DefaultAzureCredential,
        )
        from azure.storage.blob import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional Azure storage dependency")
            BlobServiceClient,
        )
        import smart_open  # noqa: PLC0415 - codex_reason("gpt5.2", "optional smart_open dependency for Azure")
    except ImportError as e:  # noqa: BLE001
        raise click.ClickException(
            f"Could not upload file to Azure Storage. Missing required packages: {e}. "
            "Please install: pip install smart-open[azure] azure-storage-blob azure-identity"
        )

    parsed = urlparse(upload_path)
    # smart_open uses 'azure://' scheme, not 'abfss://'
    # Typical forms we want to support:
    #   - abfss://container@account.dfs.core.windows.net/path
    #   - azure://container/path
    #
    # For abfss, we can derive both the container and storage account from the URL.
    # For azure://, we only know the container from the URL, so we require the
    # storage account name to be provided via environment variable.
    if parsed.scheme == "abfss":
        container_and_account = parsed.netloc
        # Example: "container@account.dfs.core.windows.net"
        container_name = container_and_account.split("@")[0]
        storage_account_name: str = container_and_account.split("@")[1].split(".")[0]
    elif parsed.scheme == "azure":
        # Example: azure://container/path
        container_name = parsed.netloc

        # We need the storage account name to construct the BlobServiceClient.
        # Require it to be passed via environment variable so users can control
        # which account is used when only given an azure:// URI.
        # Use the same standard name used elsewhere in the codebase.
        storage_account_name = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
        if not storage_account_name:
            raise click.ClickException(
                "When using an 'azure://' upload_path, you must set either "
                "AZURE_STORAGE_ACCOUNT to the Azure Storage account name."
            )
    else:
        raise click.ClickException(
            f"Unsupported Azure URI scheme for upload_path: {upload_path!r}"
        )

    # Construct azure:// URL for smart_open using the same logic as S3/GCS
    final_uploaded_filepath = f"azure://{container_name}/{object_name}"

    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=credential,
        )
        transport_params = {"client": blob_service_client}
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(
            f"Failed to create Azure BlobServiceClient: {e!r}. "
            "Please ensure Azure credentials are available (managed identity or environment variables)."
        ) from e

    try:
        with smart_open.open(
            final_uploaded_filepath, "wb", transport_params=transport_params
        ) as fout, builtins.open(file, "rb") as fin:
            fout.write(fin.read())
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(
            f"Failed to upload the working directory to Azure Storage. Error {e!r} "
            "Please validate you have exported Azure credentials with the correct write permissions. "
            "If you do not desire to upload your working directory, please set your working directory to a public remote URI or remove the runtime_environment from your job or service yaml."
        ) from e
    return final_uploaded_filepath


def _get_remote_storage_object_name(upload_path, upload_filename):
    # Strip leading slash, otherwise bucket will create a new directory called "/".
    object_name = os.path.join(urlparse(upload_path).path, upload_filename).lstrip("/")
    return object_name


def _upload_file_to_remote_storage(
    source_file: str, upload_path: str, upload_filename: str
):
    parsed_upload_path = urlparse(upload_path)
    service = parsed_upload_path.scheme
    bucket = parsed_upload_path.netloc
    object_name = _get_remote_storage_object_name(upload_path, upload_filename)
    if service in ("abfss", "azure"):
        return _upload_file_to_azure_storage(source_file, upload_path, object_name)
    if service == "s3":
        _upload_file_to_s3(source_file, bucket, object_key=object_name)
    if service == "gs":
        _upload_file_to_google_cloud_storage(
            source_file, bucket, object_name=object_name
        )
    final_uploaded_filepath = os.path.join(upload_path, upload_filename)
    try:
        from smart_open import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional smart_open dependency for remote storage")
            open,  # noqa: A004 - claude_comment("claude-opus-4-5", "smart_open.open intentionally shadows builtin for file-like API")
        )

        open(final_uploaded_filepath)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(
            f"Could not open uploaded file, maybe something went wrong while uploading: {e}."
        )

    return final_uploaded_filepath


def is_dir_remote_uri(target_dir: str) -> bool:
    parsed = urlparse(target_dir)
    return bool(parsed.scheme)


@contextmanager
def zip_local_dir(
    path: str, *, excludes: Optional[List[str]] = None
) -> Generator[Tuple[str, bytes, str], None, None]:
    """Packs the local directory into a temporary zip file.

    After the context manager exits, the file is deleted.

    Yields: (temp_path, file_contents, content_hash).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_file_path = os.path.join(temp_dir, "anyscale_generated_working_dir.zip")
        zip_directory(
            path,
            excludes=excludes if excludes is not None else [],
            output_path=zip_file_path,
            # Ray requires remote Zip URIs to consist of a single top-level directory when unzipped.
            include_parent_dir=True,
        )
        zip_file_bytes = Path(zip_file_path).read_bytes()
        yield zip_file_path, zip_file_bytes, hashlib.md5(zip_file_bytes).hexdigest()


def upload_and_rewrite_working_dir(
    runtime_env_json: Dict[str, Any],
    upload_file_to_remote_storage_fn: Callable[
        [str, str, str], str
    ] = _upload_file_to_remote_storage,
) -> Dict[str, Any]:
    """Upload a local working_dir and rewrite the working_dir field with the destination remote URI.

    After uploading, deletes the "upload_path" field because it is no longer used and is not a valid
    OSS runtime env field.
    """
    if runtime_env_json.get("working_dir") is None:
        return runtime_env_json

    working_dir = runtime_env_json["working_dir"]
    if is_dir_remote_uri(working_dir):
        # The working dir is a remote URI already
        return runtime_env_json

    upload_path = runtime_env_json["upload_path"]
    excludes = runtime_env_json.get("excludes")
    with zip_local_dir(working_dir, excludes=excludes) as (zip_file_path, _, hash_val):
        uploaded_zip_file_name = f"_anyscale_pkg_{hash_val}.zip"
        final_uploaded_filepath = upload_file_to_remote_storage_fn(
            zip_file_path, upload_path, uploaded_zip_file_name,
        )

    final_runtime_env = runtime_env_json.copy()
    final_runtime_env["working_dir"] = final_uploaded_filepath
    del final_runtime_env["upload_path"]
    return final_runtime_env


def override_runtime_env_config(
    runtime_env: Optional[Dict[str, Any]],
    anyscale_api_client: Union[DefaultApi, SDKDefaultApi],
    api_client: DefaultApi,
    workload_type: Workload,
    compute_config_id: Optional[str],
    log: BlockLogger,
) -> Optional[Dict[str, Any]]:
    """Override the working_dir, upload_path, and pip fields for a runtime_env.

    When running inside a workspace:
        1. Autopopulates the `working_dir` with the current directory.
        2. Autopopulates the `pip` field with the contents of the workspace-managed
           `requirements.txt` file.

    If the working_dir is a local path, will upload the contents to cloud storage and
    replace the field with the resulting remote URI. The upload_path can be specified
    in the runtime_env, else it will be auto-populated for the cloud.
    """

    existing_runtime_env = autopopulate_runtime_env_for_workspace(
        runtime_env=runtime_env, log=log
    )

    if not existing_runtime_env:
        return {}
    elif not existing_runtime_env.get("working_dir"):
        return existing_runtime_env

    working_dir = existing_runtime_env.get("working_dir", "")
    upload_path = existing_runtime_env.get("upload_path")

    if not is_dir_remote_uri(working_dir):
        if upload_path is not None:
            # If upload_path is specified
            # we back up the current working dir to the specified path
            new_runtime_env = upload_and_rewrite_working_dir(existing_runtime_env)
        elif is_anyscale_workspace() and "ANYSCALE_SESSION_ID" in os.environ:
            # If submitting job v2 from workspaces and no upload_path is specified,
            # we back up the current workspace content into S3
            cluster_id = os.environ["ANYSCALE_SESSION_ID"]

            decorated_cluster = api_client.get_decorated_cluster_api_v2_decorated_sessions_cluster_id_get(
                cluster_id
            ).result
            cloud_id = decorated_cluster.cloud.id

            workspace_id = os.environ["ANYSCALE_EXPERIMENTAL_WORKSPACE_ID"]

            new_runtime_env = infer_upload_path_and_rewrite_working_dir(
                api_client=api_client,
                existing_runtime_env=existing_runtime_env,
                workload_type=workload_type,
                cloud_id=cloud_id,
                log=log,
                workspace_id=workspace_id,
            )
        else:
            compute_template: ComputeTemplate = anyscale_api_client.get_compute_template(
                compute_config_id
            ).result
            cloud_id = compute_template.config.cloud_id
            new_runtime_env = infer_upload_path_and_rewrite_working_dir(
                api_client=api_client,
                existing_runtime_env=existing_runtime_env,
                workload_type=workload_type,
                cloud_id=cloud_id,
                log=log,
            )

        return new_runtime_env
    else:
        return existing_runtime_env


def parse_dot_env_file(dot_env_bytes: bytes) -> Dict[str, str]:
    """Parse a .env file and return a dictionary of key-value pairs."""
    dot_env = dot_env_bytes.split(b"\x00")
    ret = {}
    for kv in dot_env:
        if len(kv) == 0:
            # skip empty lines
            continue
        segs = kv.split(b"=", 1)
        if len(segs) == 2:
            key, value = segs
            try:
                ret[key.decode().strip()] = value.decode().strip()
            except UnicodeDecodeError:
                logger.error(f"Failed to decode env var entry: {kv!r}")
        else:
            logger.error(f"Invalid env var entry: {kv!r}")
    return ret


def parse_requirements_file(path: str) -> Optional[List[str]]:
    """TODO: add comment."""
    requirements_file = Path(path)
    if requirements_file.is_file():
        lines = requirements_file.read_text().strip().split("\n")
        parsed_requirements = []
        for line in lines:
            # Strip comments and whitespace
            hash_idx = line.find("#")
            content = (line[:hash_idx] if hash_idx != -1 else line).strip()
            if content:
                parsed_requirements.append(content)
    else:
        parsed_requirements = None

    return parsed_requirements


def autopopulate_runtime_env_for_workspace(
    runtime_env: Optional[Dict[str, Any]],
    log: BlockLogger,
    *,
    requirements_file_path: str = WORKSPACE_REQUIREMENTS_FILE_PATH,
) -> Optional[Dict[str, Any]]:
    """Autopopulates fields of the runtime_env for commands run in a workspace.

    Fields populated (if not specified by the user):
        - working_dir: set to ".".
        - pip: set to the contents of the workspace-managed requirements.txt file.
          If the file does not exist, this field will not be set.

    No-op if called outside of a workspace.
    """
    if not is_anyscale_workspace():
        return runtime_env

    if not runtime_env:
        runtime_env = {}

    if not runtime_env.get("working_dir"):
        runtime_env["working_dir"] = "."
        log.info("working_dir is not specified, using the current local directory.")

    # Workspaces maintains a `requirements.txt` file in a well-known location.
    # By default, populate the runtime environment with the contents of this file.
    # If the user passes any "pip" dependencies, do not overwrite them.
    if not is_workspace_dependency_tracking_disabled():
        parsed_requirements = parse_requirements_file(requirements_file_path)
        if parsed_requirements:
            if runtime_env.get("pip"):
                log.info(
                    "Not including workspace-tracked dependencies because "
                    "'pip' field is specified in the runtime_env."
                )
            elif runtime_env.get("conda"):
                log.info(
                    "Not including workspace-tracked dependencies because "
                    "'conda' field is specified in the runtime_env."
                )
            else:
                runtime_env["pip"] = parsed_requirements
                log.info("Including workspace pip dependencies.")

    return runtime_env


def infer_upload_path_and_rewrite_working_dir(
    *,
    api_client: DefaultApi,
    existing_runtime_env: Dict[str, Any],
    cloud_id: str,
    workload_type: Workload,
    log: BlockLogger,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Push working_dir to remote bucket and rewrite the working_dir field with the destination uri

    If the upload_path is not specified by the user, we will get the bucket name based on the cloud.
    We then rewrite the working_dir to the remote uri path
    so that the launched service will read from remote bucket directly.

    For Workspaces:
        The remote path: [s3, gs, abfss]://{bucket_name}/{org_id}/{cloud_id}/workspace_snapshots/{workspace_id}/{workload_type}/{backup_zip}
    Otherwise:
        The remote path: [s3, gs, abfss]://{bucket_name}/{org_id}/{cloud_id}/{workload_type}/{backup_zip}
        workload_type=[jobs, scheduled_jobs, services]
    """

    all_cloud_resources = api_client.get_cloud_resources_api_v2_clouds_cloud_id_resources_get(
        cloud_id=cloud_id,
    ).results

    if len(all_cloud_resources) == 0:
        raise click.ClickException(f"No cloud resources found for cloud {cloud_id}.")

    primary_cloud_resource: DecoratedCloudResource = all_cloud_resources[0]

    bucket_name = _get_cloud_storage_bucket_name_from_cloud_resource(
        primary_cloud_resource
    )

    if primary_cloud_resource.provider == CloudProviders.AWS:
        protocol = "s3"
    elif primary_cloud_resource.provider == CloudProviders.GCP:
        protocol = "gs"
    elif primary_cloud_resource.provider == CloudProviders.AZURE:
        if not bucket_name:
            raise ValueError(
                f"Cloud {cloud_id} does not have an Azure storage bucket configured. "
                "Please configure a storage bucket for this cloud."
            )

        # Parse the scheme from the bucket name (e.g., abfss://container@account.dfs.core.windows.net)
        parsed_bucket = urlparse(bucket_name)
        if not parsed_bucket.scheme:
            raise click.ClickException(
                f"Invalid Azure storage bucket name format: {bucket_name}. "
                "Expected format: abfss://container-name@storage-account.dfs.core.windows.net"
            )
        protocol = str(parsed_bucket.scheme)
        # Extract storage account name for Ray runtime environment
        # Format: abfss://container@storage-account.dfs.core.windows.net
        netloc_str = str(parsed_bucket.netloc) if parsed_bucket.netloc else ""
        if not netloc_str:
            raise click.ClickException(
                f"Invalid Azure storage bucket name format: {bucket_name}. "
                "Expected format: abfss://container-name@storage-account.dfs.core.windows.net"
            )
        storage_account_name = netloc_str.split("@")[1].split(".")[0]
        # For Azure, use netloc (container@account.dfs.core.windows.net) for path construction
        bucket_name = netloc_str
    else:
        raise click.ClickException(
            f"Currently launching jobs or services from workspaces in a {primary_cloud_resource.provider} cloud is not supported. "
            "Please contact Anyscale support for more info."
        )

    new_runtime_env = copy.deepcopy(existing_runtime_env)
    working_dir_path = Path(new_runtime_env["working_dir"]).absolute()
    log.info(f"Uploading local working_dir from '{working_dir_path}'.")

    org_id = _get_organization_id(api_client)
    if workspace_id:
        new_runtime_env[
            "upload_path"
        ] = f"{protocol}://{bucket_name}/{org_id}/{cloud_id}/workspace_snapshots/{workspace_id}/{workload_type}"
    else:
        new_runtime_env[
            "upload_path"
        ] = f"{protocol}://{bucket_name}/{org_id}/{cloud_id}/{workload_type}"

    # For Azure, set AZURE_STORAGE_ACCOUNT environment variable for Ray to download working_dir
    if primary_cloud_resource.provider == CloudProviders.AZURE and storage_account_name:
        if "env_vars" not in new_runtime_env:
            new_runtime_env["env_vars"] = {}
        new_runtime_env["env_vars"]["AZURE_STORAGE_ACCOUNT"] = storage_account_name
        log.info(
            f"Setting AZURE_STORAGE_ACCOUNT={storage_account_name} for Ray runtime environment."
        )

    new_runtime_env = upload_and_rewrite_working_dir(new_runtime_env)
    return new_runtime_env


def _get_organization_id(api_client: DefaultApi):
    user_info: UserInfo = (api_client.get_user_info_api_v2_userinfo_get().result)
    orgs = user_info.organizations
    return orgs[0].id


def _get_cloud_storage_bucket_name_from_cloud_resource(
    cloud_resource: DecoratedCloudResource,
) -> Optional[str]:
    """
    If the cloud resource has an associated AWS S3, Google Storage or Azure Storage bucket, we return its name.

    Please note that this is only for v2 clouds where customers have their own object storage.
    """
    if cloud_resource.object_storage and cloud_resource.object_storage.bucket_name:
        return cloud_resource.object_storage.bucket_name
    else:
        return None
