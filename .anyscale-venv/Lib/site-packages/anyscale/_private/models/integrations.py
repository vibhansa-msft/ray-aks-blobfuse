from dataclasses import dataclass, field
from typing import ClassVar, Dict

from anyscale._private.models.model_base import ModelBase, ModelEnum


class IntegrationType(ModelEnum):
    """Type of third-party integration for connections."""

    DATABRICKS = "DATABRICKS"

    __docstrings__: ClassVar[Dict[str, str]] = {
        DATABRICKS: "Databricks integration for Unity Catalog access",
    }


class ConnectionType:
    """Backend connection type strings."""

    DATABRICKS_U2M = "databricks_U2M"


CONNECTION_TYPE_TO_INTEGRATION_TYPE: Dict[str, IntegrationType] = {
    ConnectionType.DATABRICKS_U2M: IntegrationType.DATABRICKS,  # type: ignore[dict-item]
}


@dataclass(frozen=True)
class ConnectionConfig(ModelBase):
    """Configuration for a third-party integration connection.

    Connections allow workloads (jobs, workspaces, etc.) to access external services
    like Databricks Unity Catalog. Each connection is identified by its integration
    type and name.
    """

    __doc_py_example__ = """\
from anyscale._private.models.integrations import ConnectionConfig, IntegrationType

connection = ConnectionConfig(
    integration_type=IntegrationType.DATABRICKS,
    connection_name="my-databricks-connection",
)
"""

    __doc_yaml_example__ = """\
connections:
  - integration_type: DATABRICKS
    connection_name: my-databricks-connection
"""

    integration_type: IntegrationType = field(
        metadata={"docstring": "The type of integration (e.g., DATABRICKS)."},
    )

    def _validate_integration_type(
        self, integration_type: IntegrationType
    ) -> IntegrationType:
        if not isinstance(integration_type, IntegrationType):
            raise TypeError(
                f"'integration_type' must be an 'IntegrationType' (it is {type(integration_type)})."
            )
        return integration_type

    connection_name: str = field(
        metadata={
            "docstring": "The name of the connection as registered in the organization settings.",
        },
    )

    def _validate_connection_name(self, connection_name: str):
        if not isinstance(connection_name, str):
            raise TypeError(
                f"'connection_name' must be a string (it is {type(connection_name)})."
            )
        if not connection_name:
            raise ValueError("'connection_name' cannot be empty.")
