"""
    This module contains classes and functions for data transfer between processes.
    Communication builds on queues.
"""

class TransferFactory:
    @staticmethod
    def create(type, *args, **kwargs):
        if type == "queue":
            return QueueTransfer(*args, **kwargs)