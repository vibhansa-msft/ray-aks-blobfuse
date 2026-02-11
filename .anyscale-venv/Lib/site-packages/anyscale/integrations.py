import os
import time
from typing import Any, List, Optional

from packaging import version

from anyscale.authenticate import get_auth_api_client
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import WandBRunDetails
from anyscale.sdk.anyscale_client import JobsQuery
from anyscale.shared_anyscale_utils.util import slugify
from anyscale.shared_anyscale_utils.utils.protected_string import ProtectedString
from anyscale.util import get_endpoint
from anyscale.utils.imports.gcp import try_import_gcp_secretmanager


"""anyscale/experimental_integrations.py: Experimental util functions for W&B integration and secret store prototypes."""

WANDB_API_KEY_NAME = "WANDB_API_KEY_NAME"  # pragma: allowlist secret
WANDB_PROJECT_NAME = "WANDB_PROJECT_NAME"
WANDB_GROUP_NAME = "WANDB_GROUP_NAME"

log = BlockLogger()  # Anyscale CLI Logger


def get_aws_secret(secret_name: str, **kwargs) -> ProtectedString:
    """
    Get secret value from AWS secrets manager.

    Arguments:
        secret_name: Key of your secret
        kwargs: Optional credentials passed in to authenticate instance
    """
    import boto3  # noqa: PLC0415 - codex_reason("gpt5.2", "optional AWS SDK dependency for secrets retrieval")

    client = boto3.client("secretsmanager", **kwargs)
    response = client.get_secret_value(SecretId=secret_name)

    # Depending on whether the secret is a string or binary, one of these fields will be populated.
    if "SecretString" in response:
        secret = response.pop("SecretString")
    else:
        secret = response.pop("SecretBinary")

    return ProtectedString(secret)


def get_gcp_secret(secret_name: str, **kwargs) -> ProtectedString:
    """
    Get secret value from GCP secrets manager.

    Arguments:
        secret_name: Key of your secret
        kwargs: Optional credentials passed in to authenticate instance
    """
    from google import (  # noqa: PLC0415 - codex_reason("gpt5.2", "optional GCP SDK dependency for secrets retrieval")
        auth,
    )

    secretmanager = try_import_gcp_secretmanager()

    client = secretmanager.SecretManagerServiceClient(**kwargs)
    _, project_name = auth.default()

    name = f"projects/{project_name}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})

    return ProtectedString(response.payload.data.decode())


def wandb_get_api_key() -> ProtectedString:
    """
    Returns W&B API key based on key set in WANDB_API_KEY_NAME in
    AWS or GCP secrets manager.

    Assumes instance is running with correct IAM role, so credentials
    don't have to be passed to access secrets manager.
    """
    secret_names: List[str] = []
    wandb_api_key_name_env_var = os.environ.get(WANDB_API_KEY_NAME)
    if wandb_api_key_name_env_var:
        secret_names.append(wandb_api_key_name_env_var)
    cluster_id = os.environ.get("ANYSCALE_SESSION_ID")
    api_client = get_auth_api_client(log_output=False).api_client

    # Get cloud from cluster to use the correct method to
    # get secrets from a cloud.
    if cluster_id:
        cluster = api_client.get_decorated_cluster_api_v2_decorated_sessions_cluster_id_get(
            cluster_id
        ).result
        cloud_id = cluster.cloud_id
        cloud = api_client.get_cloud_api_v2_clouds_cloud_id_get(cloud_id).result
    else:
        raise Exception(f"Unable to find cluster {cluster_id}")

    # Store alternate secret name to try fetching in list to ensure
    # backward compatibility with old secret naming convention
    if cloud.provider == "AWS":
        secret_names.append(f"anyscale_{cloud.id}/{cluster.creator_id}/wandb_api_key")
    elif cloud.provider == "GCP":
        # GCP secret names cannot include "/"
        secret_names.append(f"anyscale_{cloud.id}-{cluster.creator_id}-wandb_api_key")
    secret_names.append(f"wandb_api_key_{cluster.creator_id}")

    if cloud.provider not in ("AWS", "GCP"):
        raise Exception(
            "The Anyscale W&B integration is currently only supported for AWS and GCP clouds."
        )
    for secret_name in secret_names:
        try:
            if cloud.provider == "AWS":
                region = cloud.region
                secret = get_aws_secret(secret_name, region_name=region)
            elif cloud.provider == "GCP":
                secret = get_gcp_secret(secret_name)
            return secret
        except Exception:  # noqa: BLE001
            log.info(f"Unable to fetch API key with name {secret_name}.")

    raise Exception("Unable to fetch API key from cloud secrets manager.")


def wandb_setup_api_key_hook() -> Optional[str]:
    """
    Returns W&B API key based on key set in WANDB_API_KEY_NAME in
    AWS or GCP secrets manager. This returns the API key in plain text,
    so take care to not save the output in any logs.

    The WANDB_SETUP_API_KEY_HOOK will point to this method so it will
    be called by the OSS WandbLoggerCallback. Because this is called
    before wandb.init(), any other setup can also be done here.
    """
    protected_api_key = wandb_get_api_key()

    try:
        import ray  # noqa: PLC0415 - codex_reason("gpt5.2", "optional Ray dependency for W&B env defaults")

        # Check if ray <= 2.2.0 before setting project and group env vars
        # because OSS will directly call the set_wandb_project_group_env_vars
        # hook for ray > 2.2.0. Dev versions of ray are greater than all other
        # versions (eg: 3.0.0.dev0).
        if version.parse(ray.__version__) <= version.parse("2.2.0"):
            # Set environment variables to define default W&B project and group.
            set_wandb_project_group_env_vars()
    except ImportError:
        # Ray should always be installed on the head node, so we should
        # never reach this case.
        log.info(
            "Unable to import `ray` so setting default W&B project and group if "
            "these parameters are not already specified."
        )
        set_wandb_project_group_env_vars()

    # API key returned in plaintext because the OSS WandbLoggerCallback
    # accepts the API key as a string arguement.
    return protected_api_key._UNSAFE_DO_NOT_USE  # noqa: SLF001


def set_wandb_project_group_env_vars():
    """
    Set WANDB_PROJECT_NAME and WANDB_GROUP_NAME environment variables
    for the OSS WandbLoggerCallback to use, based on the default mapping
    for production jobs, workspaces, and Ray jobs.
    """
    api_client = get_auth_api_client(log_output=False).api_client

    wandb_project_default = None
    wandb_group_default = None
    if os.environ.get("ANYSCALE_HA_JOB_ID"):
        production_job_id = os.environ.get("ANYSCALE_HA_JOB_ID")
        production_job = api_client.get_job_api_v2_decorated_ha_jobs_production_job_id_get(
            production_job_id
        ).result

        wandb_project_default = "anyscale_default_project"
        wandb_group_default = slugify(production_job.name)
    elif os.environ.get("ANYSCALE_EXPERIMENTAL_WORKSPACE_ID"):
        workspace_id = os.environ.get("ANYSCALE_EXPERIMENTAL_WORKSPACE_ID")
        workspace = api_client.get_workspace_api_v2_experimental_workspaces_workspace_id_get(
            workspace_id
        ).result

        wandb_project_default = slugify(workspace.name)
    elif _try_get_ray_job_id():
        ray_job_id = _try_get_ray_job_id()
        cluster_id = os.environ.get("ANYSCALE_SESSION_ID")
        if cluster_id:
            cluster_name = api_client.get_session_api_v2_sessions_session_id_get(
                cluster_id
            ).result.name
        else:
            cluster_name = "anyscale_default_project"
        wandb_project_default = slugify(cluster_name)
        wandb_group_default = slugify(f"ray_job_id_{ray_job_id}")

    if not os.environ.get(WANDB_PROJECT_NAME) and wandb_project_default:
        os.environ[WANDB_PROJECT_NAME] = wandb_project_default
    if not os.environ.get(WANDB_GROUP_NAME) and wandb_group_default:
        os.environ[WANDB_GROUP_NAME] = wandb_group_default


def wandb_send_run_info_hook(run: Any) -> None:
    """
    The WANDB_PROCESS_RUN_INFO points to this method and is called on
    the `run` output of `wandb.init()`.

    Send the W&B URL to the control plane, and populate the link back to
    Anyscale from the W&B run config.
    """
    auth_api_client = get_auth_api_client(log_output=False)
    api_client = auth_api_client.api_client
    anyscale_api_client = auth_api_client.anyscale_api_client

    try:
        import wandb  # noqa: PLC0415 - codex_reason("gpt5.2", "optional W&B dependency for run reporting")
    except ImportError:
        raise Exception("Unable to import wandb.")

    assert isinstance(
        run, wandb.sdk.wandb_run.Run
    ), "`run` argument must be of type wandb.sdk.wandb_run.Run"

    if os.environ.get("ANYSCALE_HA_JOB_ID"):
        production_job_id = os.environ.get("ANYSCALE_HA_JOB_ID")
        production_job = api_client.get_job_api_v2_decorated_ha_jobs_production_job_id_get(
            production_job_id
        ).result
        # Wait up to 5 sec to ensure the job run exists for the production job. We have previously
        # observed a delay in populating job runs in the DB when /api/snapshot has large
        # response in the old cluster snapshot service code. This should no longer
        # be an issue for the events stream stack.
        retry = 5
        while not production_job.last_job_run_id and retry > 0:
            time.sleep(1)
            production_job = api_client.get_job_api_v2_decorated_ha_jobs_production_job_id_get(
                production_job_id
            ).result
            retry -= 1
        if not production_job.last_job_run_id:
            log.info(
                "Unable to find latest job execution for this production Anyscale job."
            )
        api_client.put_production_job_wandb_run_details_api_v2_integrations_production_job_wandb_run_details_production_job_id_put(
            production_job_id=production_job_id,
            wand_b_run_details=WandBRunDetails(
                wandb_project_url=run.get_project_url(), wandb_group=run.group
            ),
        )
        run.config.anyscale_logs = get_endpoint(f"/jobs/{production_job_id}")
    elif os.environ.get("ANYSCALE_EXPERIMENTAL_WORKSPACE_ID"):
        workspace_id = os.environ.get("ANYSCALE_EXPERIMENTAL_WORKSPACE_ID")
        api_client.put_workspace_wandb_run_details_api_v2_integrations_workspace_wandb_run_details_workspace_id_put(
            workspace_id=workspace_id,
            wand_b_run_details=WandBRunDetails(
                wandb_project_url=run.get_project_url(), wandb_group=run.group
            ),
        )
        cluster_id = os.environ.get("ANYSCALE_SESSION_ID")
        if cluster_id:
            run.config.anyscale_logs = get_endpoint(
                f"/workspaces/{workspace_id}/{cluster_id}"
            )
    elif _try_get_ray_job_id():
        ray_job_id = _try_get_ray_job_id()
        cluster_id = os.environ.get("ANYSCALE_SESSION_ID")
        # TODO(nikita): Use ray.runtime_context.get_runtime_context().get_job_submission_id() instead of
        # API call. This will be available with Ray 2.3: https://github.com/ray-project/ray/issues/28089
        ray_job_resp = anyscale_api_client.search_jobs(
            JobsQuery(cluster_id=cluster_id, ray_job_id=ray_job_id)
        ).results
        # Wait up to 5 sec to send W&B url to control plane in case there
        # is delay populating ray job in DB.
        # We have previously observed this delay when /api/snapshot has large
        # response in the old cluster snapshot service code, but this should no longer
        # be an issue for the events stream stack.
        retry = 5
        while not len(ray_job_resp) and retry > 0:
            time.sleep(1)
            ray_job = anyscale_api_client.search_jobs(
                JobsQuery(cluster_id=cluster_id, ray_job_id=ray_job_id)
            ).results
            retry -= 1
        if len(ray_job_resp):
            ray_job = ray_job_resp[0]
            api_client.put_job_wandb_run_details_api_v2_integrations_job_wandb_run_details_job_id_put(
                job_id=ray_job.id,
                wand_b_run_details=WandBRunDetails(
                    wandb_project_url=run.get_project_url(), wandb_group=run.group
                ),
            )
            run.config.anyscale_logs = get_endpoint(
                f"/interactive-sessions/{ray_job.id}"
            )
        else:
            log.info("Unable to find Ray job in Anyscale to populate with W&B URL.")


def _try_get_ray_job_id():
    """
    Get the Ray job id using the Ray API.
    """
    try:
        import ray  # noqa: PLC0415 - codex_reason("gpt5.2", "optional Ray dependency for job context")

        return ray.get_runtime_context().get_job_id()
    except Exception:  # noqa: BLE001
        # Alternatively get job id from environment variable
        # if error importing Ray. This should never happen because
        # this code is called from a hook in Ray.
        return os.environ.get("RAY_JOB_ID")
