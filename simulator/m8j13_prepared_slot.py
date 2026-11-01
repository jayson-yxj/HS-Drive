"""One pending CPU item, independent of the active serial inference."""
from queue import Queue,Full
import threading

class PreparedSlot:
    def __init__(self):
        self.queue=Queue(1);self.permit=threading.Semaphore(1);self.closed=threading.Event()
    def reserve(self):return not self.closed.is_set() and self.permit.acquire(timeout=.02)
    def cancel(self):self.permit.release()
    def publish(self,item):
        while not self.closed.is_set():
            try:self.queue.put(item,timeout=.02);return True
            except Full:pass
        return False
    def take(self,timeout=.02):
        item=self.queue.get(timeout=timeout);self.permit.release();return item
    def empty(self):return self.queue.empty()
    def close(self):self.closed.set()
