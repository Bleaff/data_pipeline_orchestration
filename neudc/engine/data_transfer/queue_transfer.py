import sys
import queue
from collections import deque

import zmq
from multiprocessing.util import register_after_fork

from .queue_base import QueueLike

class ZeroQueue(QueueLike):
    def __init__(self, port=None):
        self.port = port
        self.context = zmq.Context()
        self.socket_pub = self.context.socket(zmq.PUB)
        if not self.port:
            self.port = self.socket_pub.bind_to_random_port(f"tcp://*")
        else:
            self.socket_pub.bind(f"tcp://*:{self.port}")

        self.socket_sub = self.context.socket(zmq.SUB)
        self.socket_sub.connect(f"tcp://localhost:{self.port}")
        self.socket_sub.subscribe("")

        self.poller = zmq.Poller()
        self.poller.register(self.socket_sub, zmq.POLLIN)

        if sys.platform != 'win32':
            register_after_fork(self, ZeroQueue._after_fork)

    def __str__(self):
        return f"{self.__class__.__name__}(port={self.port})"

    def get(self, timeout: float|int|None = None):
        timeout_millis = int(timeout * 1000) if timeout else None

        if self.socket_sub in dict(self.poller.poll(timeout=timeout_millis)):
            return self.get_nowait()

    def get_nowait(self):
        return self.socket_sub.recv_pyobj(zmq.NOBLOCK)

    def put(self, item):
        self.socket_pub.send_pyobj(item)

    def put_nowait(self, item):
        self.socket_sub.send_pyobj(item, zmq.NOBLOCK)

    def _after_fork(self):
        self._reset()

    def _reset(self):
        self.socket_pub = self.context.socket(zmq.PUB)
        self.socket_pub.connect(f"tcp://*:{self.port}")

        self.socket_sub = self.context.socket(zmq.SUB)
        self.socket_sub.connect(f"tcp://localhost:{self.port}")
        self.socket_sub.subscribe("")

        self.poller = zmq.Poller()
        self.poller.register(self.socket_sub, zmq.POLLIN)

    def __getstate__(self):
        print("__getstate__")
        return (self.socket_pub, self.socket_sub, self.poller)

    def __setstate__(self, state):
        print("__setstate__")
        (self.socket_pub, self.socket_sub, self.poller) = state


class ZeroQueueConsumer(QueueLike):
    def __init__(self, port=None):
        self.port = port
        self.context = zmq.Context()
        self.socket_sub = self.context.socket(zmq.REP)
        if not self.port:
            self.port = self.socket_sub.bind_to_random_port(f"tcp://*")
        else:
            self.socket_sub.connect(f"tcp://localhost:{self.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket_sub, zmq.POLLIN)

    def __str__(self):
        return f"{self.__class__.__name__}(port={self.port})"

    def __del__(self):
        self.socket_sub.close()
        self.context.term()

    def get(self, timeout=None):
        if timeout:
            timeout = timeout * 1000
        if self.socket_sub in dict(self.poller.poll(timeout=timeout)):
            data = self.socket_sub.recv_pyobj(zmq.NOBLOCK)
            self.socket_sub.send(b"0", zmq.NOBLOCK)
            return data
        
    def get_nowait(self):
        if self.socket_sub in dict(self.poller.poll(timeout=1)): # ?
            try:
                data = self.socket_sub.recv_pyobj(zmq.NOBLOCK)
                self.socket_sub.send(b"0", zmq.NOBLOCK)
            except zmq.ZMQError:
                raise queue.Empty
            return data
        else:
            raise queue.Empty

    def put(self, item):
        raise NotImplementedError
    
    def put_nowait(self, item):
        raise NotImplementedError


class ZeroQueueProducer(QueueLike):
    DEQUE_LEN = 100
    TIMEOUT_MS = 100
    TMP_N = 40

    def __init__(self, port=None, deque_len=None, dlq_db_path=""):      
        self.port = port
        if deque_len:
            self.deque = deque(maxlen=deque_len)
        else:
            self.deque = deque(maxlen=self.DEQUE_LEN)
        self.init()

    def __str__(self):
        return f"{self.__class__.__name__}(port={self.port})"

    def init(self):
        self.context = zmq.Context()

        self.socket_pub = self.context.socket(zmq.REQ)
        self.socket_pub.setsockopt(zmq.LINGER, 0)
        if not self.port:
            self.port = self.socket_pub.bind_to_random_port(f"tcp://*")
        else:
            self.socket_pub.connect(f"tcp://localhost:{self.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket_pub, zmq.POLLIN)

    def __del__(self):
        self.dlq.close()
        self.stop()

    def stop(self):
        self.socket_pub.close()
        self.context.term()

    def _put_from_deque(self, timeout=TIMEOUT_MS):
        _item = self.deque[0]
        self.socket_pub.send_pyobj(_item)
        if self.socket_pub in dict(self.poller.poll(timeout=timeout)):
            self.socket_pub.recv()
            self.deque.popleft()
            return True
        else:
            return False

    def put(self, item, timeout: float|int|None = None):
        if len(self.deque) == self.DEQUE_LEN:
            old_item = self.deque.popleft()
            self.dlq.save_to_dlq_raw(old_item)
        self.deque.append(item)
        if timeout:
            timeout_millis = int(timeout * 1000)
            while len(self.deque):
                if not self._put_from_deque(timeout_millis):
                    self.stop()
                    self.init()
                    break
        else:
            while len(self.deque):
                if not self._put_from_deque():
                    self.stop()
                    self.init()
                    break
                    
    def put_nowait(self, _item):
        self.socket_pub.send_pyobj(_item)
        if self.socket_pub in dict(self.poller.poll(timeout=self.TIMEOUT_MS//5+1)):
            self.socket_pub.recv()
        else:
            self.dlq.save_to_dlq_raw(_item)
            self.stop()
            self.init()
            raise queue.Full


    def get(self, timeout):
        raise NotImplementedError
    
    def get_nowait(self):
        raise NotImplementedError
