import click


def try_import_ray():
    try:
        import ray  # noqa: PLC0415 - codex_reason("gpt5.2", "optional Ray dependency for CLI commands")

        return ray
    except ImportError:
        raise click.ClickException(
            "Ray not installed locally on this machine but required "
            "for the command. Please install with `pip install 'anyscale[all]'`."
        )
