from neudc.engine.reader.base import AReaderThread
from threading import Thread

class ImageReader(AReaderThread):
    def __init__(self):
        super().__init__()
    
    