"""
Commands to interact with the Anyscale Sessions API
"""

from typing import Optional

import click

from anyscale.authenticate import get_auth_api_client
from anyscale.client.openapi_client.models import StartSessionOptions
from anyscale.formatters import common_formatter


@click.group(
    "sessions", help="Commands to interact with the Sessions API.",
)
def sessions() -> None:
    pass


@sessions.command(
    name="list", short_help="Lists all Sessions belonging to the Project."
)
@click.argument("project_id", required=True)
@click.option(
    "--count", type=int, default=10, help="Number of projects to show. Defaults to 10."
)
@click.option(
    "--paging-token",
    required=False,
    help="Paging token used to fetch subsequent pages of projects.",
)
def list_sessions(project_id: str, count: int, paging_token: Optional[str],) -> None:
    """Lists all the non-deleted sessions under PROJECT_ID. """

    api_client = get_auth_api_client().anyscale_api_client
    response = api_client.list_sessions(
        project_id, count=count, paging_token=paging_token
    )

    print(common_formatter.prettify_json(response.to_dict()))


@sessions.command(name="get", short_help="Retrieves a Session.")
@click.argument("session_id", required=True)
def get_session(session_id: str) -> None:
    """Get details about the Session with id SESSION_ID"""

    api_client = get_auth_api_client().anyscale_api_client
    response = api_client.get_session(session_id)

    print(common_formatter.prettify_json(response.to_dict()))


@sessions.command(name="delete", short_help="Deletes a Session.")
@click.argument("session_id", required=True)
def delete_session(session_id: str) -> None:
    """Delete the Session with id SESSION_ID"""

    api_client = get_auth_api_client().anyscale_api_client
    api_client.delete_session(session_id)


@sessions.command(name="start", short_help="Sets session goal state to Running.")
@click.argument("session_id", required=True)
@click.argument("cluster_config", required=False)
def start_session(session_id: str, cluster_config: Optional[str]) -> None:
    """Sets session goal state to Running
    using the given cluster_config (if specified).
    A session with goal state running will eventually
    transition from its current state to Running, or
    remain Running if its current state is already Running.
    Retrieves the corresponding session operation.
    The session will update if necessary to the given
    cluster_config.

    cluster_config: New cluster config to apply to the Session.
    """

    api_client = get_auth_api_client().anyscale_api_client
    start_session_options = StartSessionOptions(cluster_config=cluster_config)
    response = api_client.start_session(session_id, start_session_options)
    print(common_formatter.prettify_json(response.to_dict()))
