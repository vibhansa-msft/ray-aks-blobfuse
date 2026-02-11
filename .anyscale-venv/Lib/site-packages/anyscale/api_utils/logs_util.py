import asyncio
import re
from typing import List, Optional

import aiohttp
import requests

from anyscale.shared_anyscale_utils.utils.asyncio import gather_in_batches


CLUSTER_CONNECT_TIMEOUT = 30


async def _download_logs_concurrently(
    log_chunk_urls: List[str], parallelism: int, bearer_token: Optional[str] = None
) -> str:
    logs_across_chunks: List[str] = await gather_in_batches(  # type: ignore
        parallelism,
        *[
            _download_log_from_s3_url(url, bearer_token=bearer_token)
            for url in log_chunk_urls
        ],
    )
    logs_across_chunks = [log.strip() for log in logs_across_chunks]
    return "\n".join(logs_across_chunks)


async def _download_log_from_ray_json_response(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        response = await asyncio.wait_for(
            session.get(url), timeout=CLUSTER_CONNECT_TIMEOUT
        )
        logs: str = (await response.json()).get("logs", "")
        return logs


async def _download_log_from_s3_url(
    url: str, bearer_token: Optional[str] = None,
) -> str:
    # Note that the URL is presigned, so no token needs to be passed in the request
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
        async with session.get(url, headers=headers) as response:
            return await response.text()


def _download_log_from_s3_url_sync(
    url: str, bearer_token: Optional[str] = None,
) -> str:
    # Note that the URL is presigned, so no token needs to be passed in the request
    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def _remove_ansi_escape_sequences(s: str) -> str:
    # Required as the log may contain ANSI escape sequeneces (e.g. for coloring in the terminal)
    # Regex pattern from https://stackoverflow.com/a/14693789
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", s)
