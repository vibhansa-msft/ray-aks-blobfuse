import logging

import requests
from typing_extensions import Literal

from anyscale.anyscale_pydantic import BaseModel
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as BaseApi
from anyscale.util import get_cluster_model_for_current_workspace


logger = logging.getLogger(__name__)


WORKSPACE_NOTIFICATION_ADDRESS = "http://localhost:8266/simplefileserver/notifications"


class WorkspaceNotificationAction(BaseModel):
    type: Literal["navigate-service", "navigate-workspace-tab", "navigate-external-url"]
    title: str
    value: str


class WorkspaceNotification(BaseModel):
    body: str
    action: WorkspaceNotificationAction


def send_workspace_notification(
    anyscale_api_client: BaseApi, notification: WorkspaceNotification
):

    try:
        if get_cluster_model_for_current_workspace(anyscale_api_client) is None:
            # We are not in a workspace, so we don't want to send a notification
            return
        requests.post(WORKSPACE_NOTIFICATION_ADDRESS, json=notification.dict())
    except Exception:
        logger.exception("Failed to send notification to UI")
