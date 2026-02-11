from typing import List, Optional

from anyscale.client.openapi_client.models import LogFileChunk
from anyscale.client.openapi_client.models.node_type import NodeType


class LogGroupFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.chunks: List[LogFileChunk] = []

    def insert_chunk(self, chunk: LogFileChunk):
        self.chunks.append(chunk)

    def get_chunks(self, reverse: bool = False) -> List[LogFileChunk]:
        if reverse:
            return sorted(self.chunks, key=lambda chunk: chunk.chunk_name, reverse=True)
        else:
            return sorted(self.chunks, key=lambda chunk: chunk.chunk_name)

    def get_name(self) -> str:
        return self.file_name

    def get_target_path(self) -> str:
        assert len(self.chunks) >= 1
        sample_chunk = self.chunks[0]
        node_type = "head" if sample_chunk.node_type == NodeType.HEAD_NODE else "worker"

        # We have special-case handling here to deal with non-standard log files that are sourced from something that's
        # a single node (e.g. webterminal logs + job driver logs do not have node information associated with them):

        path = "/".join(["logs", sample_chunk.cluster_id, sample_chunk.session_id])

        # If the logs came from a specific node, add node information.
        if sample_chunk.node_ip and sample_chunk.instance_id:
            path = (
                path
                + "/"
                + f"{node_type}-{sample_chunk.node_ip}-{sample_chunk.instance_id}"
            )

        # Add the rest of the destination filename (this might be something like "dashboard.log" or "serve/health.log").
        path = path + "/" + sample_chunk.file_name

        return path

    def get_size(self) -> int:
        return sum([chunk.size for chunk in self.chunks])


class LogGroupNode:
    def __init__(self, node_ip: str, instance_id: str, node_type: NodeType):
        self.node_ip = node_ip
        self.instance_id = instance_id
        self.node_type = node_type
        self.files: List[LogGroupFile] = []

    def insert_chunk(self, chunk: LogFileChunk):
        for file in self.files:
            if file.file_name == chunk.file_name:
                file.insert_chunk(chunk)
                return
        file = LogGroupFile(file_name=chunk.file_name)
        file.insert_chunk(chunk)
        self.files.append(file)

    def get_files(self) -> List[LogGroupFile]:
        return sorted(self.files, key=lambda file: file.file_name)

    def get_chunks(self) -> List[LogFileChunk]:
        result = []
        for file in self.files:
            result.extend(file.get_chunks())
        return result


class LogGroupSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.nodes: List[LogGroupNode] = []

    def insert_chunk(self, chunk: LogFileChunk):
        for node in self.nodes:
            if node.node_ip == chunk.node_ip and node.instance_id == chunk.instance_id:
                node.insert_chunk(chunk)
                return
        node = LogGroupNode(
            node_ip=chunk.node_ip,
            instance_id=chunk.instance_id,
            node_type=chunk.node_type,
        )
        node.insert_chunk(chunk)
        self.nodes.append(node)

    def get_nodes(self) -> List[LogGroupNode]:
        return sorted(
            self.nodes,
            key=lambda node: (node.node_type, node.node_ip, node.instance_id),
        )

    def get_files(self) -> List[LogGroupFile]:
        result = []
        for node in self.nodes:
            result.extend(node.get_files())
        return result

    def get_chunks(self) -> List[LogFileChunk]:
        result = []
        for node in self.nodes:
            result.extend(node.get_chunks())
        return result


class LogGroup:
    def __init__(self, bearer_token: Optional[str] = None):
        self.sessions: List[LogGroupSession] = []
        self.bearer_token = bearer_token

    def insert_chunk(self, chunk: LogFileChunk):
        for session in self.sessions:
            if session.session_id == chunk.session_id:
                session.insert_chunk(chunk)
                return
        session = LogGroupSession(session_id=chunk.session_id)
        session.insert_chunk(chunk)
        self.sessions.append(session)

    def get_sessions(self) -> List[LogGroupSession]:
        return sorted(self.sessions, key=lambda session: session.session_id)

    def get_files(self) -> List[LogGroupFile]:
        result = []
        for session in self.sessions:
            result.extend(session.get_files())
        return result

    def get_chunks(self) -> List[LogFileChunk]:
        result = []
        for session in self.sessions:
            result.extend(session.get_chunks())
        return result
