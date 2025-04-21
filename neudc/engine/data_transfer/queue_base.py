from abc import ABC, abstractmethod

class QueueLike(ABC):

    @abstractmethod
    def put(self, message):
        raise NotImplementedError

    @abstractmethod
    def put_nowait(self, message):
        raise NotImplementedError

    @abstractmethod
    def get(self, timeout):
        raise NotImplementedError

    @abstractmethod
    def get_nowait(self):
        raise NotImplementedError
