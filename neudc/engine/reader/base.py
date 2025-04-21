from abc import ABC, abstractmethod

from neudc.engine.data_transfer.queue_base import QueueLike


class AReaderThread(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.to: list[int] = kwargs.get("to", None)
        self.out_queue: list[QueueLike] = None

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def status(self) -> bool:
        pass

    def put(self, message) -> None:
        for queue in self.out_queue:
            pass
