import dataclasses
import enum
import ipaddress
import socket
from typing import List, Optional, Set, Union

from anyscale.cli_logger import CloudSetupLogger
from anyscale.client.openapi_client.models import CloudAnalyticsEventCloudResource
from anyscale.utils.cloud_utils import CloudSetupError


_SOCKET_TO_PROTO_MAP = {
    getattr(socket, attr): attr.replace("IPPROTO_", "").lower()
    for attr in dir(socket)
    if attr.startswith("IPPROTO_")
}

IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


@dataclasses.dataclass
class CapacityThreshold:
    min_network_size: IPNetwork
    warn_network_size: IPNetwork
    resource_type: str
    cloud_resource: CloudAnalyticsEventCloudResource

    def verify_network_capacity(
        self, *, cidr_block_str: str, resource_name: str, logger: CloudSetupLogger
    ) -> bool:
        cidr_block = ipaddress.ip_network(cidr_block_str, strict=False)

        min_hosts = self.min_network_size.num_addresses
        warn_hosts = self.warn_network_size.num_addresses

        if cidr_block.num_addresses < min_hosts:
            logger.error(
                f"The provided {self.resource_type} ({resource_name})'s CIDR block ({cidr_block}) is too"
                f" small. We want at least {min_hosts} addresses,"
                f" but this {self.resource_type} only has {cidr_block.num_addresses}. Please reach out to"
                f" support if this is an issue!"
            )
            logger.log_resource_error(
                self.cloud_resource, CloudSetupError.CIDR_BLOCK_TOO_SMALL
            )
            return False
        elif cidr_block.num_addresses < warn_hosts:
            logger.warning(
                f"The provided {self.resource_type} ({resource_name})'s CIDR block ({cidr_block}) is probably"
                f" too small. We suggest at least {warn_hosts}"
                f" addresses, but this {self.resource_type} only supports up to"
                f" {cidr_block.num_addresses} addresses."
            )

        return True


AWS_VPC_CAPACITY = CapacityThreshold(
    min_network_size=ipaddress.ip_network("10.0.0.0/24"),
    warn_network_size=ipaddress.ip_network("10.0.0.0/20"),
    resource_type="VPC",
    cloud_resource=CloudAnalyticsEventCloudResource.AWS_VPC,
)

AWS_SUBNET_CAPACITY = CapacityThreshold(
    min_network_size=ipaddress.ip_network("10.0.0.0/28"),
    warn_network_size=ipaddress.ip_network("10.0.0.0/24"),
    resource_type="Subnet",
    cloud_resource=CloudAnalyticsEventCloudResource.AWS_SUBNET,
)

GCP_SUBNET_CAPACITY = dataclasses.replace(
    AWS_VPC_CAPACITY,
    resource_type="Subnet",
    cloud_resource=CloudAnalyticsEventCloudResource.GCP_SUBNET,
)


class Direction(enum.Enum):
    INGRESS = "INGRESS"
    EGRESS = "EGRESS"


class Protocol(enum.Enum):
    all = "all"
    tcp = "tcp"
    udp = "udp"
    icmp = "icmp"
    other = "other"

    @classmethod
    def from_val(cls, val: Union[str, int]):
        # Convert protocol numbers to strings
        if isinstance(val, int) or val.isdigit():
            val = _SOCKET_TO_PROTO_MAP.get(int(val), "other")

        if hasattr(cls, val.lower()):  # type: ignore
            return cls(val.lower())  # type: ignore

        return Protocol.other


@dataclasses.dataclass
class FirewallRule:
    direction: Direction
    protocol: Protocol
    network: IPNetwork
    ports: Optional[Set[int]]

    def __repr__(self):
        return f"FirewallRule(direction='{self.direction.value}', protocol='{self.protocol.value}', network='{self.network}', num_ports={len(self.ports)})"

    def _canonicalize_ports(self):
        if self.ports is None or len(self.ports) == 0:
            self.ports = set(range(1, 65536))
            return
        ports = self.ports
        self.ports = set()
        for port in ports:
            if "-" in str(port):
                lower, upper = port.split("-")
                self.ports.update(range(int(lower), int(upper) + 1))
            else:
                self.ports.add(int(port))

    def __post_init__(self):
        self.direction = Direction(self.direction)

        self._canonicalize_ports()


def check_inbound_firewall_permissions(
    rules: List[FirewallRule],
    protocol: Protocol,
    ports: Optional[Set[int]],
    source_range: IPNetwork,
    check_ingress_source_range: bool = True,
) -> bool:
    ports_remaining = ports.copy() if ports is not None else set(range(1, 65536))
    for rule in rules:

        if (
            rule.direction != Direction.INGRESS
            or not isinstance(source_range, type(rule.network))
            or not source_range.subnet_of(rule.network)  # type: ignore
        ) and check_ingress_source_range:
            continue

        if rule.protocol in (Protocol.all, protocol):
            ports_remaining.difference_update(rule.ports or {})

    # Port 0 is not a thing, so just account for it being weird
    return len(ports_remaining) == 0 or ports_remaining == {0}
