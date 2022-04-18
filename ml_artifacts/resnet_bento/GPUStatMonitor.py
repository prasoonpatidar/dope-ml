import GPUtil
from threading import Thread
import time


class GPUStatMonitor(Thread):
    def __init__(self, in_queue, delay=0.01):
        super(GPUStatMonitor, self).__init__()
        self.is_running = False
        self.in_queue = in_queue
        self.gpus = GPUtil.getGPUs()
        self.delay=delay

    def run(self):
        while self.is_running:
            self.in_queue.put([gpu_device.load for gpu_device in self.gpus])
            # time.sleep(self.delay)

    def start(self):
        self.is_running=True
        super(GPUStatMonitor, self).start()

    def stop(self):
        self.is_running = False
        super(GPUStatMonitor, self).stop()

def get_all_queue_result(queue):
    result_list = []
    while not queue.empty():
        result_list.append(queue.get())

    return result_list