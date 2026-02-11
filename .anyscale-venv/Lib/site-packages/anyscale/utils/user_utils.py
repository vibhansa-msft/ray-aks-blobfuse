from click import ClickException

from anyscale.client.openapi_client.api.default_api import DefaultApi
from anyscale.client.openapi_client.models.collaborator_type import CollaboratorType


def get_user_id_by_email(api_client: DefaultApi, email: str) -> str:
    users = api_client.list_organization_collaborators_api_v2_organization_collaborators_get(
        email=email, collaborator_type=CollaboratorType.ONLY_USER_ACCOUNTS,
    ).results
    if len(users) == 0:
        raise ClickException(f"No user found with email {email}.")

    if len(users) > 1:
        raise ClickException(
            f"Multiple users found with email {email}. Please provide a unique email."
        )
    return users[0].user_id
