from abc import ABC, abstractmethod
import time
from typing import Callable, Iterator, Optional


class Timer(ABC):
    @abstractmethod
    def poll(self, *, timeout_s: float, interval_s: float) -> Iterator[None]:
        """Yield every interval_s until the timeout.

        Exits when timeout_s is reached. DOES NOT raise TimeoutError.
        """


class RealTimer(Timer):
    def poll(self, *, timeout_s: float, interval_s: float) -> Iterator[None]:
        start_time_s = time.time()
        while True:
            yield

            if time.time() - start_time_s >= timeout_s:
                break
            else:
                time.sleep(interval_s)


class FakeTimer(Timer):
    def __init__(self):
        self._on_poll_iteration: Optional[Callable[[int], None]] = None

    def set_on_poll_iteration(self, on_poll_iteration: Callable[[int], None]):
        self._on_poll_iteration = on_poll_iteration

    def poll(self, *, timeout_s: float, interval_s: float) -> Iterator[None]:
        iteration: int = 0
        elapsed_time_s: float = 0
        while True:
            if self._on_poll_iteration is not None:
                self._on_poll_iteration(iteration)
            iteration += 1

            yield

            elapsed_time_s += interval_s
            if elapsed_time_s >= timeout_s:
                break
