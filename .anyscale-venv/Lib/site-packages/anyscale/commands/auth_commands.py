import click

from anyscale.controllers.auth_controller import AuthController


@click.group(
    "auth",
    short_help="Configure the Anyscale authentication credentials",
    help="""Configure the Anyscale authentication credentials""",
    hidden=True,
)
def auth_cli() -> None:
    pass


@auth_cli.command(name="set", help="Set up credentials and save it to a file")
def auth_set() -> None:
    auth_controller = AuthController()
    auth_controller.set()


@auth_cli.command(
    name="show", help="Show the information of the authenticated user using credentials"
)
def auth_show() -> None:
    auth_controller = AuthController()
    auth_controller.show()


@auth_cli.command(
    name="fix", help="Fix the permission of the existing credentials file"
)
def auth_fix() -> None:
    auth_controller = AuthController()
    auth_controller.fix()


@auth_cli.command(name="remove", help="Remove the current credentials file")
def auth_remove() -> None:
    auth_controller = AuthController()
    auth_controller.remove()
