from io import BufferedReader
import os

from rich.progress import (
    BarColumn,
    FileSizeColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TotalFileSizeColumn,
)
from rich.style import Style
from rich.table import Column

from anyscale.shared_anyscale_utils.bytes_util import Bytes


class FileDownloadProgress(Progress):
    """
    Returns a progress table for SDK logs / CLI output that looks like:

    ```
    0:00:01  50% [=========>---------]  1.5 MB / 3 MB  Downloading 'file.txt'
    ```

    (`Downloading 'file.txt'` is the task description)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            # Columns
            TimeElapsedColumn(),  #     1. Time elapsed
            TaskProgressColumn(),  #    2. Percent complete
            BarColumn(),  #             3. Horizontal progress bar
            FileSizeColumn(),  #        4. {uploaded}/{total} filesize
            "/",
            TotalFileSizeColumn(),
            TextColumn(  #              5. Current step
                "{task.description}",
                table_column=Column(width=50),
                style=Style(bold=True, color="blue"),
            ),
            *args,
            **kwargs
        )


class ProgressFileReader:
    """
    A file reader that updates a progress table (ie. the number of bytes read) as it reads the file.

    This is required because:
    - S3 presigned upload URLs upload the file in one go
    - Thus, we use the file read position as a proxy for the upload progress
    """

    def __init__(
        self, file: BufferedReader, progress: Progress, task_id: TaskID,
    ):
        self.file = file
        self.progress = progress
        self.task_id = task_id
        self.bytes_read = 0
        self.file_size = os.path.getsize(file.name)

    def read(self, size=-1):
        chunk = self.file.read(size)
        if chunk:
            self.bytes_read += len(chunk)
            self.progress.update(self.task_id, advance=len(chunk))
        return chunk

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.read(Bytes.KB)
        if not chunk:
            raise StopIteration
        return chunk

    def __len__(self):
        return self.file_size
