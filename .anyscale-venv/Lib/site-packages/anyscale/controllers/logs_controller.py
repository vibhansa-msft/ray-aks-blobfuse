import asyncio
from collections import OrderedDict
from datetime import timedelta
import errno
import json
import os
import shutil
import tempfile
from typing import Any, List, Optional, Tuple

import aiohttp
import click
from rich.console import Console
from rich.progress import track

from anyscale.api_utils.job_util import _get_job_run_id
from anyscale.cli_logger import BlockLogger
from anyscale.client.openapi_client.models import (
    LogDownloadConfig,
    LogDownloadRequest,
    LogDownloadResult,
    LogFileChunk,
    LogFilter,
)
from anyscale.controllers.base_controller import BaseController
from anyscale.sdk.anyscale_client.models.cluster import Cluster
from anyscale.sdk.anyscale_client.models.session_state import SessionState
from anyscale.utils.logs_utils import LogGroup


# Page size for listing bucket
DEFAULT_PAGE_SIZE = 1000

# Timeout for API Requests
DEFAULT_TIMEOUT = 30

# Timeout for Downloading Files
DEFAULT_READ_TIMEOUT = 30

DEFAULT_TTL = 14400
DEFAULT_PARALLELISM = 10

# Most users should have be able to handle this many open file descriptors,
# but add ability to override just in case.
DEFAULT_UNPACK_MAX_FILE_DESCRIPTORS = int(
    os.environ.get("ANYSCALE_LOGS_UNPACK_MAX_FILE_DESCRIPTORS", "100")
)


class LogsController(BaseController):
    def __init__(
        self, log: Optional[BlockLogger] = None, initialize_auth_api_client: bool = True
    ):
        if log is None:
            log = BlockLogger()

        super().__init__(initialize_auth_api_client=initialize_auth_api_client)
        self.log = log
        self.console = Console()
        # Track if we've already sent the unpack log message, so that we don't
        # spam the CLI output when users have had multiple sessions.
        self._unpack_log_sent = False

    def get_cluster_id_for_last_prodjob_run(self, prodjob_id: str):
        last_job_run_id = _get_job_run_id(self.anyscale_api_client, job_id=prodjob_id)
        last_job_run = self.api_client.get_decorated_job_api_v2_decorated_jobs_job_id_get(
            job_id=last_job_run_id
        )
        job_run = last_job_run.result
        return job_run.cluster_id

    def get_cluster_id_for_workspace(self, workspace_id: str):
        try:
            workspace = self.api_client.get_workspace_api_v2_experimental_workspaces_workspace_id_get(
                workspace_id
            )
            if workspace is None or workspace.result is None:
                raise click.ClickException(f"Workspace {workspace_id} not found.")
            workspace_result = workspace.result
            if (
                not hasattr(workspace_result, "cluster_id")
                or workspace_result.cluster_id is None
            ):
                raise click.ClickException(
                    f"Workspace {workspace_id} does not have an associated cluster."
                )
            return workspace_result.cluster_id
        except Exception as e:  # noqa: BLE001
            if isinstance(e, click.ClickException):
                raise
            # Handle 404 errors specifically
            if hasattr(e, "status") and e.status == 404:
                raise click.ClickException(
                    f"Workspace {workspace_id} not found."
                ) from e
            raise click.ClickException(
                f"Failed to get workspace {workspace_id}: {e!s}"
            ) from e

    def get_cluster_id_for_service(  # noqa: C901, PLR0912
        self, service_id: str, version_name_or_id: Optional[str] = None
    ):
        service = self.api_client.get_service_api_v2_services_v2_service_id_get(
            service_id=service_id
        )
        if service is None or service.result is None:
            raise click.ClickException(f"Service {service_id} not found.")

        # For services, we need to get the cluster_id from the service's version
        # Services can have multiple clusters, so we'll get the specified version or latest
        try:
            versions = self.api_client.get_service_versions_api_v2_services_v2_service_id_versions_get(
                service_id=service_id
            )
            if not versions or not versions.results:
                raise click.ClickException(
                    f"Service {service_id} has no versions available."
                )

            # Find the specified version or use the latest
            target_version = None
            if version_name_or_id:
                # Try to find by ID first (full ID or truncated ID)
                for v in versions.results:
                    if (
                        hasattr(v, "id")
                        and v.id
                        and (
                            v.id == version_name_or_id
                            or v.id.endswith(version_name_or_id)
                        )
                    ):
                        target_version = v
                        break
                # If not found by ID, try by name
                if target_version is None:
                    for v in versions.results:
                        if (
                            hasattr(v, "version")
                            and v.version
                            and v.version == version_name_or_id
                        ):
                            target_version = v
                            break
                if target_version is None:
                    raise click.ClickException(
                        f"Service version '{version_name_or_id}' not found for service {service_id}. "
                        f"Available versions: {', '.join([v.version if hasattr(v, 'version') and v.version else v.id[-10:] if hasattr(v, 'id') and v.id else 'unknown' for v in versions.results[:5]])}"
                    )
            else:
                # Get the latest version (first in the list is typically the latest)
                # Prefer RUNNING versions, but fall back to any version if none are running
                running_versions = [
                    v
                    for v in versions.results
                    if hasattr(v, "current_state")
                    and v.current_state
                    and str(v.current_state).upper() == "RUNNING"
                ]
                target_version = (
                    running_versions[0] if running_versions else versions.results[0]
                )

            # Print information about which service version was fetched
            version_name = (
                getattr(target_version, "version", None)
                or getattr(target_version, "id", None)
                or "unknown"
            )
            version_state = getattr(target_version, "current_state", None) or "unknown"

            def print_version_info(cluster_id: str):
                self.console.print(
                    f"Fetching logs for service version: {version_name} "
                    f"(Cluster ID: {cluster_id}, State: {version_state})"
                )

            # Try to get cluster_id from the version's production_job_ids
            # For services, production_job_ids contains production job IDs (which may be service IDs)
            if (
                hasattr(target_version, "production_job_ids")
                and target_version.production_job_ids
                and len(target_version.production_job_ids) > 0
            ):
                # Try each production_job_id to find the cluster_id
                # (for services, there may be multiple replicas/jobs per version)
                for production_job_id in target_version.production_job_ids:
                    try:
                        # Get the decorated production job to access cluster_id
                        job = self.api_client.get_job_api_v2_decorated_ha_jobs_production_job_id_get(
                            production_job_id=production_job_id
                        )
                        if job and job.result and job.result.state:
                            # Try cluster_id from state first
                            if (
                                hasattr(job.result.state, "cluster_id")
                                and job.result.state.cluster_id
                            ):
                                print_version_info(job.result.state.cluster_id)
                                return job.result.state.cluster_id
                            # Fall back to cluster.id if cluster object exists
                            if (
                                hasattr(job.result.state, "cluster")
                                and job.result.state.cluster
                                and hasattr(job.result.state.cluster, "id")
                                and job.result.state.cluster.id
                            ):
                                print_version_info(job.result.state.cluster_id)
                                return job.result.state.cluster.id
                    except Exception:  # noqa: BLE001
                        # Continue to next production_job_id if this one fails
                        continue
            # If we can't get it from production_job_ids, try to get from the version directly
            if hasattr(target_version, "cluster_id") and target_version.cluster_id:
                print_version_info(job.result.state.cluster_id)
                return target_version.cluster_id
        except click.ClickException:
            # Re-raise click exceptions (like version not found)
            raise
        except Exception:  # noqa: BLE001
            # If we can't get versions, fall through to error message
            pass
        # If we can't determine the cluster, provide a helpful error
        raise click.ClickException(
            f"Unable to determine cluster ID for service {service_id}"
            + (f" version '{version_name_or_id}'" if version_name_or_id else "")
            + ". "
            "Services may have multiple clusters. "
            "Please use 'anyscale logs cluster --id <cluster_id>' with a specific cluster ID, "
            "or view the service in the UI to find the cluster ID."
        )

    def render_logs(
        self, log_group: LogGroup, parallelism: int, read_timeout: timedelta, tail: int
    ):
        self._download_or_stdout(
            log_group=log_group,
            parallelism=parallelism,
            read_timeout=read_timeout,
            write_to_stdout=True,
            tail=tail,
            show_progress_bar=False,
        )

    def get_log_group(
        self,
        filter: LogFilter,  # noqa: A002
        page_size: Optional[int],
        ttl_seconds: Optional[int],
        timeout: timedelta,
    ) -> LogGroup:
        if filter.cluster_id:
            cluster: Cluster = self.anyscale_api_client.get_cluster(
                filter.cluster_id
            ).result
            if cluster.state == SessionState.RUNNING:
                self.log.warning(
                    "The latest 24 hours of logs are not guaranteed if the cluster is still running. "
                    "If so, please view logs via the Anyscale UI or Ray dashboard."
                )
        chunks, bearer_token = self._list_log_chunks(
            log_filter=filter,
            page_size=page_size,
            ttl_seconds=ttl_seconds,
            timeout=timeout,
        )
        log_group = self._group_log_chunk_list(chunks, bearer_token)
        return log_group

    def download_logs(  # noqa: PLR0913
        self,
        # Provide filters
        filter: LogFilter,  # noqa: A002
        # List files config
        page_size: int = DEFAULT_PAGE_SIZE,
        ttl_seconds: int = DEFAULT_TTL,
        timeout: timedelta = timedelta(seconds=DEFAULT_TIMEOUT),
        read_timeout: timedelta = timedelta(seconds=DEFAULT_READ_TIMEOUT),
        # Download config
        parallelism: int = DEFAULT_PARALLELISM,
        download_dir: Optional[str] = None,
        write_to_stdout: bool = False,
        unpack: bool = True,
        resource_id: Optional[str] = None,
    ) -> None:
        log_group = self.get_log_group(filter, page_size, ttl_seconds, timeout)
        self.console.log(
            f"Discovered {len(log_group.get_files())} log files across {len(log_group.get_chunks())} chunks."
        )

        self._download_or_stdout(
            download_dir=download_dir,
            read_timeout=read_timeout,
            parallelism=parallelism,
            log_group=log_group,
            show_progress_bar=True,
            write_to_stdout=write_to_stdout,
            unpack=unpack,
            resource_id=resource_id,
        )

    def _resolve_download_directory(
        self,
        write_to_stdout: bool,
        download_dir: Optional[str],
        resource_id: Optional[str],
    ) -> Optional[str]:
        """Determine the final download directory based on options."""
        if write_to_stdout:
            return None

        base_dir = download_dir or os.getcwd()
        if resource_id:
            return os.path.join(base_dir, resource_id)
        return base_dir

    def _write_logs_to_stdout(
        self, log_group: LogGroup, tmp_dir: str, tail: int
    ) -> bool:
        """Write logs to stdout. Returns True if we hit the tail limit and should stop."""
        lines_read = 0
        for log_file in log_group.get_files():
            is_tail = tail > 0
            chunks = log_file.get_chunks(reverse=is_tail)
            for chunk in chunks:
                with open(
                    os.path.join(tmp_dir, chunk.chunk_name), errors="ignore"
                ) as source:
                    if tail > 0:
                        # Tail is enabled, so read log lines in reverse.
                        # TODO (shomilj): Make this more efficient (don't read everything into memory before reversing it)
                        # For now, this is fine (the chunks are <10 MB, so this isn't that inefficient).
                        for line in reversed(source.readlines()):
                            print(line.strip())
                            lines_read += 1
                            if lines_read >= tail:
                                return True
                    else:
                        print(source.read())
        return False

    def _write_logs_to_files(
        self, log_group: LogGroup, tmp_dir: str, final_dir: str, unpack: bool
    ) -> None:
        """Write logs to files in the final directory."""
        for log_file in log_group.get_files():
            chunks = log_file.get_chunks(reverse=False)
            real_path = os.path.join(final_dir, log_file.get_target_path())
            real_dir = os.path.dirname(real_path)
            if not os.path.exists(real_dir):
                os.makedirs(real_dir)

            chunks_written = 0
            with open(real_path, "w") as dest:
                for chunk in chunks:
                    downloaded_chunk_path = os.path.join(tmp_dir, chunk.chunk_name)
                    if not os.path.exists(downloaded_chunk_path):
                        self.log.error(
                            "Download failed for file: %s", chunk.chunk_name,
                        )
                        continue
                    with open(downloaded_chunk_path, errors="ignore") as source:
                        for line in source:
                            dest.write(line)
                        dest.write("\n")
                    chunks_written += 1

            if unpack:
                self._unpack_structured_log(real_path)

            if chunks_written == 0:
                os.remove(real_path)

    def _download_or_stdout(  # noqa: PLR0913
        self,
        log_group: LogGroup,
        parallelism: int,
        read_timeout: timedelta,
        download_dir: Optional[str] = None,
        write_to_stdout: bool = False,
        tail: int = -1,
        show_progress_bar: bool = False,
        unpack: bool = True,
        resource_id: Optional[str] = None,
    ):
        if len(log_group.get_chunks()) == 0:
            return

        try:
            final_download_dir = self._resolve_download_directory(
                write_to_stdout, download_dir, resource_id
            )

            # Create target directory before temp dir so temp is on the same partition.
            # This ensures disk space errors reference the actual target location.
            if final_download_dir and not os.path.exists(final_download_dir):
                os.makedirs(final_download_dir)

            with tempfile.TemporaryDirectory(dir=final_download_dir) as tmp_dir:
                # Download all files to a temporary directory.
                asyncio.run(
                    # TODO (shomilj): Add efficient tailing method here.
                    self._download_files(
                        base_folder=tmp_dir,
                        log_chunks=log_group.get_chunks(),
                        bearer_token=log_group.bearer_token,
                        parallelism=parallelism,
                        read_timeout=read_timeout,
                        show_progress_bar=show_progress_bar,
                    )
                )

                if write_to_stdout:
                    if self._write_logs_to_stdout(log_group, tmp_dir, tail):
                        return
                else:
                    assert final_download_dir is not None
                    self._write_logs_to_files(
                        log_group, tmp_dir, final_download_dir, unpack
                    )
                    absolute_path = os.path.abspath(final_download_dir)
                    self.console.log(
                        f"Download complete! Files have been downloaded to {absolute_path}"
                    )
        except OSError as err:
            if err.errno == errno.EMFILE:
                raise click.ClickException(
                    "Too many open files. Try doubling your open files limit by running \n"
                    "ulimit -n $(($(ulimit -n) * 2))"
                )
            raise

    def _group_log_chunk_list(
        self, chunks: List[LogFileChunk], bearer_token: Optional[str] = None
    ) -> LogGroup:
        # This has to happen locally because it happens after we retrieve all file metadata through the paginated
        # backend API for listing S3/GCS buckets.
        group = LogGroup(bearer_token)
        for chunk in chunks:
            group.insert_chunk(chunk=chunk)
        return group

    def _list_log_chunks(
        self,
        log_filter: LogFilter,
        page_size: Optional[int],
        ttl_seconds: Optional[int],
        timeout: timedelta,
    ) -> Tuple[List[LogFileChunk], Optional[str]]:
        next_page_token: Optional[str] = None
        all_log_chunks: List[LogFileChunk] = []
        bearer_token = None

        with self.console.status("Scanning available logs...") as status:
            while True:
                request = LogDownloadRequest(
                    filter=log_filter,
                    config=LogDownloadConfig(
                        next_page_token=next_page_token,
                        page_size=page_size,
                        ttl_seconds=ttl_seconds,
                        use_bearer_token=True,
                    ),
                )
                result: LogDownloadResult = self.api_client.get_log_files_api_v2_logs_get_log_files_post(
                    log_download_request=request, _request_timeout=timeout
                ).result
                bearer_token = result.bearer_token
                all_log_chunks.extend(result.log_chunks)
                if status:
                    status.update(
                        f"Scanning available logs...discovered {len(all_log_chunks)} log file chunks."
                    )
                if (
                    result.next_page_token is None
                    or result.next_page_token == next_page_token
                ):
                    break
                next_page_token = result.next_page_token

        return all_log_chunks, bearer_token

    async def _download_file(  # noqa: PLR0913
        self,
        sem: asyncio.Semaphore,
        pos: int,  # noqa: ARG002
        file_name: str,
        url: str,
        size: int,
        session: aiohttp.ClientSession,
        read_timeout: timedelta,
        bearer_token: Optional[str] = None,
    ) -> None:
        async with sem:
            download_dir = os.path.dirname(file_name)
            if download_dir and not os.path.exists(download_dir):
                os.makedirs(download_dir)

            timeout = aiohttp.ClientTimeout(
                total=None, sock_connect=30, sock_read=read_timeout.seconds
            )
            headers = (
                {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
            )
            async with session.get(url, timeout=timeout, headers=headers) as response:
                if response.status == 200:
                    try:
                        with open(file_name, "wb") as fhand:
                            async for chunk in response.content.iter_chunked(1024):
                                fhand.write(chunk)
                    except OSError as err:
                        if err.errno == errno.ENOSPC:
                            error_dir = os.path.abspath(download_dir or os.getcwd())
                            disk_usage = shutil.disk_usage(error_dir)
                            available_bytes = disk_usage.free
                            required_bytes = size or 0
                            delta_bytes = max(required_bytes - available_bytes, 0)
                            message = (
                                "Not enough disk space while downloading logs. "
                                f"Target directory: {error_dir}. "
                                f"Attempting to write {self._format_bytes(required_bytes)} but "
                                f"only {self._format_bytes(available_bytes)} is available."
                            )
                            if delta_bytes > 0:
                                message += (
                                    " Free up approximately "
                                    f"{self._format_bytes(delta_bytes)} and try again."
                                )
                            raise click.ClickException(message) from err
                        raise
                else:
                    self.log.error(
                        f"Unable to download file {file_name}! response: [{response.status}, {await response.text()}]"
                    )

    async def _download_files(
        self,
        base_folder: Optional[str],
        log_chunks: List[LogFileChunk],
        parallelism: int,
        read_timeout: timedelta,
        show_progress_bar: bool = False,
        bearer_token: Optional[str] = None,
    ) -> List[str]:
        sem = asyncio.Semaphore(parallelism)
        downloads = []
        connector = aiohttp.TCPConnector(limit_per_host=parallelism)
        paths = []
        async with aiohttp.ClientSession(connector=connector) as session:
            for pos, log_chunk in enumerate(log_chunks):
                path = os.path.join(base_folder or "", log_chunk.chunk_name.lstrip("/"))
                paths.append(path)
                downloads.append(
                    asyncio.create_task(
                        self._download_file(
                            sem,
                            pos,
                            path,
                            log_chunk.chunk_url,
                            log_chunk.size,
                            session,
                            read_timeout,
                            bearer_token=bearer_token,
                        )
                    )
                )

            if show_progress_bar:
                for task in track(
                    asyncio.as_completed(downloads),
                    description="Downloading...",
                    total=len(downloads),
                    transient=True,
                ):
                    await task
            else:
                await asyncio.gather(*downloads)

        return paths

    @staticmethod
    def _format_bytes(num: int) -> str:
        value = float(num)
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        for unit in units:
            if value < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(value)} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} PB"

    def _unpack_structured_log(self, log_path: str):
        directory = os.path.dirname(log_path)
        fname = os.path.basename(log_path)
        if fname != "combined-worker.log":
            return
        if not self._unpack_log_sent:
            self.console.log("Unpacking `combined-worker.log` into separate files.")
            self._unpack_log_sent = True
        error_count = 0
        seen = set()
        with _FileDescriptorCache(DEFAULT_UNPACK_MAX_FILE_DESCRIPTORS) as fds, open(
            log_path
        ) as f:
            for line in f.readlines():
                try:
                    j = json.loads(line)
                    msg = j["message"]
                    filename = os.path.join(directory, j["filename"])

                    # Track files we've seen. The first time we see a file,
                    # we should use "wt" to overwrite previous content, for
                    # example if a user uses the download command multiple
                    # times. On subsequent times, we use "at" to append to
                    # existing content, in case the file descriptor has been
                    # closed in the meantime.
                    if filename in seen:
                        mode = "at"
                    else:
                        seen.add(filename)
                        mode = "wt"

                    out_f = fds.get(filename, mode)
                    out_f.write(f"{msg}\n")
                except KeyboardInterrupt:
                    raise
                except OSError as err:
                    if err.errno == errno.EMFILE:
                        raise
                    error_count += 1
                except Exception:  # noqa: BLE001
                    # Soft fail on unpacking the combined log. Worst case scenario,
                    # users can still parse the combined log manually
                    error_count += 1
        if error_count:
            self.log.warning(
                f"{error_count} error(s) while unpacking {fname}. Check combined-worker.log "
                "for the full contents of worker logs."
            )


class _FileDescriptorCache:
    """
    LRU Cache for file descriptors to use while uncombining worker logs, since
    opening and closing a file descriptor per line would be slow.
    """

    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        # Recently used entries are near the "end"
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def __enter__(self) -> "_FileDescriptorCache":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, tb: Any):
        for f in self.cache.values():
            try:
                f.close()
            except Exception:  # noqa: BLE001
                # There's not much we can do if we fail to close a file descriptor,
                # just move on to the next one.
                continue

    def get(self, filename: str, mode: str):
        if filename in self.cache:
            # Update position in cache
            self.cache.move_to_end(filename)
            return self.cache[filename]
        if len(self.cache) >= self.max_entries:
            # Pop and close file descriptor closest to the "front"
            _, f = self.cache.popitem(last=False)
            f.close()
        self.cache[filename] = open(filename, mode)  # noqa: SIM115
        return self.cache[filename]
