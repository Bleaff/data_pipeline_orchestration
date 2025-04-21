"""
    This module contains classes and functions for data transfer between processes.
    Communication builds on queues.
"""

from neudc.engine.data_transfer.queue_base import QueueLike
from neudc.engine.data_transfer.queue_transfer import QueueConsumer, QueueProducer, ZeroQueue
class TransferFactory:
    @staticmethod
    def create(data_type, *args, **kwargs) -> QueueLike:
        if data_type == "Queue":
            return ZeroQueue(*args, **kwargs)
        elif data_type == "QueueProducer":
            return QueueProducer(*args, **kwargs)
        elif data_type == "QueueConsumer":
            return QueueConsumer(*args, **kwargs)
        else:
            raise ValueError(f"Unknown transfer type: {data_type}")
