import json
import logging
import os
from typing import Any

import click

from anyscale.cli_logger import BlockLogger
from anyscale.commands.aggregated_instance_usage_commands import (
    aggregated_instance_usage_cli,
)
from anyscale.commands.anyscale_api.api_commands import anyscale_api
from anyscale.commands.auth_commands import auth_cli
from anyscale.commands.cloud_commands import cloud_cli
from anyscale.commands.cluster_commands import cluster_cli
from anyscale.commands.compute_config_commands import compute_config_cli
from anyscale.commands.config_commands import config_cli
from anyscale.commands.experimental_integrations_commands import (
    experimental_integrations_cli,
)
from anyscale.commands.image_commands import image_cli
from anyscale.commands.job_commands import job_cli
from anyscale.commands.job_queue_commands import job_queue_cli
from anyscale.commands.login_commands import anyscale_login, anyscale_logout
from anyscale.commands.logs_commands import log_cli
from anyscale.commands.machine_commands import machine_cli
from anyscale.commands.machine_pool_commands import machine_pool_cli
from anyscale.commands.migrate_commands import migrate_cli
from anyscale.commands.organization_invitation_commands import (
    organization_invitation_cli,
)
from anyscale.commands.policy_commands import policy_cli
from anyscale.commands.project_commands import project_cli
from anyscale.commands.resource_quota_commands import resource_quota_cli
from anyscale.commands.schedule_commands import schedule_cli
from anyscale.commands.scim_commands import scim_cli
from anyscale.commands.service_account_commands import service_account_cli
from anyscale.commands.service_commands import service_cli
from anyscale.commands.session_commands_hidden import session_cli
from anyscale.commands.user_commands import user_cli
from anyscale.commands.user_group_commands import user_group_cli
from anyscale.commands.workspace_commands_v2 import workspace_cli as workspace_cli_v2
import anyscale.conf
import anyscale.telemetry  # IMPORTANT: auto-patches click instrumentation on import
from anyscale.utils.cli_version_check_util import log_warning_if_version_needs_upgrade


logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.CRITICAL)

log = BlockLogger()  # CLI Logger

if anyscale.conf.AWS_PROFILE is not None:
    logger.info("Using AWS profile %s", anyscale.conf.AWS_PROFILE)
    os.environ["AWS_PROFILE"] = anyscale.conf.AWS_PROFILE


class AliasedGroup(click.Group):
    # This is from https://stackoverflow.com/questions/46641928/python-click-multiple-command-names
    def get_command(self, ctx: Any, cmd_name: str) -> Any:
        if cmd_name in ALIASES:
            cmd_name = ALIASES[cmd_name].name
        return super().get_command(ctx, cmd_name)


@click.group(
    "anyscale",
    invoke_without_command=True,
    no_args_is_help=True,
    cls=AliasedGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--version",
    "-v",
    "version_flag",
    is_flag=True,
    default=False,
    help="Current anyscale version.",
)
@click.option(
    "--json",
    "show_json",
    is_flag=True,
    default=False,
    help="Return output as json, for use with --version.",
)
@click.pass_context
def cli(ctx: Any, version_flag: bool, show_json: bool) -> None:
    if version_flag:
        ctx.invoke(version_cli, show_json=show_json)
    log_warning_if_version_needs_upgrade()


@click.command(name="version", help="Display version of the anyscale CLI.")
@click.option(
    "--json", "show_json", is_flag=True, default=False, help="Return output as json."
)
def version_cli(show_json: bool) -> None:
    if show_json:
        print(json.dumps({"version": anyscale.__version__}))
    else:
        print(anyscale.__version__)


@cli.command(
    name="help", help="Display help documentation for anyscale CLI.", hidden=True
)
@click.pass_context
def anyscale_help(ctx: Any) -> None:
    print(ctx.parent.get_help())


cli.add_command(session_cli)
cli.add_command(cloud_cli)
cli.add_command(config_cli)
cli.add_command(migrate_cli)
cli.add_command(project_cli)
cli.add_command(version_cli)
cli.add_command(job_cli)
cli.add_command(job_queue_cli)
cli.add_command(schedule_cli)
cli.add_command(service_cli)
cli.add_command(cluster_cli)
cli.add_command(workspace_cli_v2)
cli.add_command(experimental_integrations_cli)
cli.add_command(auth_cli)

cli.add_command(anyscale_help)
cli.add_command(compute_config_cli)
cli.add_command(image_cli)

# Commands to interact with the Anyscale API
cli.add_command(anyscale_api)

cli.add_command(log_cli)
cli.add_command(anyscale_login)
cli.add_command(anyscale_logout)
cli.add_command(machine_cli)
cli.add_command(machine_pool_cli)
cli.add_command(service_account_cli)
cli.add_command(resource_quota_cli)
cli.add_command(aggregated_instance_usage_cli)
cli.add_command(user_cli)
cli.add_command(organization_invitation_cli)
cli.add_command(user_group_cli)
cli.add_command(scim_cli)
cli.add_command(policy_cli)

ALIASES = {
    "h": anyscale_help,
    "schedules": schedule_cli,
    "jobs": job_cli,
    "jq": job_queue_cli,
    "services": service_cli,
    "cluster-compute": compute_config_cli,
    "images": image_cli,
    "sa": service_account_cli,
}


def main() -> Any:
    return cli()


if __name__ == "__main__":
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(os.environ.get("ANYSCALE_LOGLEVEL", "WARN"))
    main()
