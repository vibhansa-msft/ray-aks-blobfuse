from typing import Optional

from rich.console import Console

from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import (
    DeleteMachineRequest,
    ListMachinesResponse,
)
from anyscale.controllers.base_controller import BaseController


class MachineController(BaseController):
    def __init__(
        self, log: Optional[BlockLogger] = None, initialize_auth_api_client: bool = True
    ):
        if log is None:
            log = BlockLogger()

        super().__init__(initialize_auth_api_client=initialize_auth_api_client)
        self.log = log
        self.console = Console()

    def list_machines(self, machine_pool_name: str,) -> ListMachinesResponse:
        response: ListMachinesResponse = self.api_client.list_machines_api_v2_machines_get(
            machine_pool_name=machine_pool_name,
        ).result
        return response

    def delete_machine(
        self, machine_id: str, machine_pool_name: Optional[str],
    ):
        self.api_client.delete_machine_api_v2_machines_delete_post(
            DeleteMachineRequest(
                machine_id=machine_id, machine_pool_name=machine_pool_name,
            ),
        )
