from typing import Optional

import click

import anyscale
from anyscale.aggregated_instance_usage.models import DownloadCSVFilters
from anyscale.cli_logger import BlockLogger
from anyscale.commands import command_examples
from anyscale.commands.util import AnyscaleCommand


log = BlockLogger()  # CLI Logger


@click.group("aggregated-instance-usage", help="Access aggregated instance usage.")
def aggregated_instance_usage_cli() -> None:
    pass


@aggregated_instance_usage_cli.command(
    name="download-csv",
    cls=AnyscaleCommand,
    example=command_examples.AGGREGATED_INSTANCE_USAGE_DOWNLOAD_CSV_EXAMPLE,
)
@click.option(
    "--start-date",
    required=True,
    type=str,
    help="The start date (inclusive) of the aggregated instance usage report. Format: YYYY-MM-DD",
)
@click.option(
    "--end-date",
    required=True,
    type=str,
    help="The end date (inclusive) of the aggregated instance usage report. Format: YYYY-MM-DD",
)
@click.option(
    "--cloud",
    required=False,
    type=str,
    help="The name of the cloud to filter the report by.",
)
@click.option(
    "--project",
    required=False,
    type=str,
    help="The name of the project to filter the report by.",
)
@click.option(
    "--directory",
    required=False,
    type=str,
    help="The directory to save the CSV file to. Default is the current directory.",
)
@click.option(
    "--hide-progress-bar",
    required=False,
    type=bool,
    help="Whether to hide the progress bar. Default is False.",
)
def download_csv(
    start_date: str,
    end_date: str,
    cloud: Optional[str],
    project: Optional[str],
    directory: Optional[str],
    hide_progress_bar: Optional[bool] = False,
) -> None:
    """
    Download an aggregated instance usage report as a zipped CSV to the provided directory.
    """
    log.info("Downloading aggregated instance usage CSV...")
    try:
        anyscale.aggregated_instance_usage.download_csv(
            DownloadCSVFilters(
                start_date=start_date,
                end_date=end_date,
                cloud=cloud,
                project=project,
                directory=directory,
                hide_progress_bar=hide_progress_bar,
            )
        )
    except ValueError as e:
        log.error(f"{e}")
        return
