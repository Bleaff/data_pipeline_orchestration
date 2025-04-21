from abc import ABC, abstractmethod
from typing import List
from zmq import 
class AReaderThread(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.to :List[int] = kwargs.get('to', None)
        self.out_queue List[]= None
    
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

        
