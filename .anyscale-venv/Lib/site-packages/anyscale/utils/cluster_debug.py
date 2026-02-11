from dataclasses import dataclass
import socket
import ssl
from typing import Callable, Dict, Iterator, Optional

import requests

from anyscale.cli_logger import BlockLogger


@dataclass
class DebugTestResult:
    name: str
    success: bool
    error: Optional[Exception] = None
    warnings: Optional[str] = None


def debug_cluster(
    logger: BlockLogger, cluster_domain_name: str, anyscale_ip: str
) -> Iterator[DebugTestResult]:
    has_experienced_a_failure: bool = False

    tests = list(_tests_for(cluster_domain_name, anyscale_ip).items())

    for name, test in tests:
        try:
            warning = test()

            yield DebugTestResult(name, True, warnings=warning)
        except Exception as e:  # noqa: BLE001
            yield DebugTestResult(name, False, e)

            if not has_experienced_a_failure:
                has_experienced_a_failure = True

                try:
                    cluster_ip_addr = socket.gethostbyname(cluster_domain_name)
                except Exception as e:  # noqa: BLE001
                    resp = requests.get(
                        "https://dns.google/resolve",
                        params={"name": cluster_domain_name},
                    )
                    resp.raise_for_status()

                    resp_json = resp.json()
                    cluster_ip_addr = resp_json["Answer"][0]["data"]
                logger.warning(
                    "Failure detected, retrying tests by directly communicating with the IP address."
                )
                tests.extend(
                    _tests_for(
                        cluster_ip_addr, anyscale_ip, tls_sni=cluster_domain_name
                    ).items()
                )


def _tests_for(
    endpoint: str, anyscale_ip: str, tls_sni: Optional[str] = None
) -> Dict[str, Callable[[], Optional[str]]]:
    """Attempt to debug networking for an Anyscale Cluster.

    Arguments:
        endpoint: Hostname or IP address of the cluster
        tls_sni: Hostname to use for TLS SNI (used if used to directly dial the IP address)

    Also ask for the IP from the control Plane

    Fallbacks:
    1. Try directly using different IPs + hostname

    """
    return {
        **(
            {
                "dns-lookup": lambda: _dns_test(endpoint, anyscale_ip),
                "dns-over-https": lambda: _dns_over_https_test(endpoint, anyscale_ip),
            }
            if tls_sni is None
            else {}
        ),
        "tcp-connect": lambda: _tcp_test(endpoint),
        "tls-no-sni": lambda: _tls_no_sni(endpoint),
        "tls-connect": lambda: _tls_test(endpoint, tls_sni),
        "https-get": lambda: _https_test(endpoint, tls_sni),
    }


def _dns_test(endpoint: str, anyscale_ip: str) -> Optional[str]:
    resolved_ip = socket.gethostbyname(endpoint)
    if resolved_ip != anyscale_ip:
        return (
            f"Unexpected local DNS resolution:\nLocal DNS resolves this to {resolved_ip}\n"
            f"Anyscale expects this host to resolve to {anyscale_ip}\n"
            "This is safe to ignore if you are using a VPN to connect to your cluster.\n"
        )
    return None


def _dns_over_https_test(endpoint: str, anyscale_ip: str) -> Optional[str]:
    resp = requests.get("https://dns.google/resolve", params={"name": endpoint})
    resp.raise_for_status()

    resp_json = resp.json()
    answers = resp_json.get("Answer", [])
    assert len(answers) > 0, "Got no answers for a successful query"

    resolved_ips = {answer["data"] for answer in answers}
    if anyscale_ip not in resolved_ips:
        return (
            f"Unexpected Public DNS resolution. This should never happen\n"
            f"`8.8.8.8` resolves this host to {resolved_ips}\n"
            f"Anyscale expects this host to resolve to {anyscale_ip}\n"
        )
    return None


def _tcp_test(endpoint: str) -> None:
    with socket.create_connection((endpoint, 443)):
        pass


def _tls_test(endpoint: str, hostname: Optional[str] = None) -> None:
    """Create a TLS connection to the destination.

        The optional hostname is used if the TLS connection is made directly to an IP address
        and we want to set the SNI separately.
    """
    if hostname is None:
        hostname = endpoint

    with socket.create_connection(
        (endpoint, 443)
    ) as sock, ssl.create_default_context().wrap_socket(sock, server_hostname=hostname):
        pass


def _tls_no_sni(endpoint: str) -> None:
    """Create a TLS connection to the destination without specifying an SNI.

    This is used to test if middleboxes/firewalls are rejecting connections based on SNI.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with socket.create_connection((endpoint, 443)) as sock, ctx.wrap_socket(
        sock, server_hostname=None
    ):
        pass


def _https_test(endpoint: str, hostname: Optional[str] = None) -> None:
    if hostname is None:
        hostname = endpoint

    session = requests.Session()
    session.mount(f"https://{endpoint}", HostnameSpecificAdapter(hostname))
    resp = session.get(f"https://{endpoint}")

    if resp.status_code != requests.codes["unauthorized"]:
        raise RuntimeError(
            f"Unexpected status code from https://{endpoint}: {resp.status_code}"
        )

    assert (
        "x-anyscale-authentication-status" in resp.headers
        and resp.headers["x-anyscale-authentication-status"] == "Unauthorized"
    ), "Failed to get an expected anyscale-unique header from the response."


class HostnameSpecificAdapter(requests.adapters.HTTPAdapter):
    """Adapter to handle directly connecting to an IP address with a specific TLS Hostname.

    This allows requests to be made to https://<ip address>, and still have TLS work for the server name.
    """

    def __init__(self, hostname: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class SingleHostSSLContext(ssl.SSLContext):
            def wrap_socket(self, *args, **kwargs):
                kwargs["server_hostname"] = hostname
                return super().wrap_socket(*args, **kwargs)

        self.init_poolmanager(
            self._pool_connections,  # type:ignore
            self._pool_maxsize,  # type:ignore
            ssl_context=SingleHostSSLContext(),
            assert_hostname=hostname,
        )
