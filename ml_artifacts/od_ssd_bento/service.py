'''Main service file for object detection model'''
import numpy as np
from PIL import Image as PILImage
import bentoml
from bentoml.io import Image, NumpyNdarray, JSON
import torch, torch.nn
from torchvision import transforms
import time

from queue import Queue
import psutil
from GPUStatMonitor import GPUStatMonitor, get_all_queue_result
import GPUtil

#setup stat monitor
gpus = GPUtil.getGPUs()
gpu_stat_queue = Queue()


# Load the model
od_runner = bentoml.pytorch.load_runner('object_detection_ssd:latest')
model = bentoml.pytorch.load('object_detection_ssd:latest')

# create new service
od_svc = bentoml.Service('object_detection',runners=[od_runner])

# preprocess image
def od_preprocess(image):
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    return image

# main api for detection
@od_svc.api(
    input=NumpyNdarray(),
    output= JSON(),
)
def detect(np_input_image):
    # stats monitoring code
    psutil.cpu_percent(interval=0.1)
    memory_usage_pre = psutil.virtual_memory()
    if len(gpus) > 0:
        gpu_mem_pre = [gpu_device.memoryUtil for gpu_device in gpus]
        gpu_m = GPUStatMonitor(gpu_stat_queue)
        gpu_m.start()
    else:
        gpu_mem_pre = []

    # actual runtime
    start_time = time.time()
    input_tensor = od_preprocess(np.uint8(np_input_image))

    # move the input and model to GPU for speed if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",DEVICE)
    input_tensor = input_tensor.to(DEVICE)
    od_model = model.to(DEVICE)

    # stats monitoring code
    inf_time = time.time() - start_time
    total_cpu_utilization = psutil.cpu_percent(interval=None)
    if len(gpus) > 0:
        gpu_mem_post = [gpu_device.memoryUtil for gpu_device in gpus]
        gpu_m.stop()
        gpu_load = get_all_queue_result(gpu_stat_queue)
    else:
        gpu_mem_post,gpu_load = [], []
    memory_usage_post = psutil.virtual_memory()

    # with torch.no_grad():
    output = od_model(input_tensor)[0]
    result = {
        'boxes':output['boxes'].cpu().detach().numpy().tolist(),
        'labels':output['labels'].cpu().detach().numpy().tolist(),
        'scores':output['scores'].cpu().detach().numpy().tolist(),
        'time':inf_time,
        'cpu_util':total_cpu_utilization,
        'cpu_times':psutil.cpu_percent(interval=inf_time, percpu=True),
        'ram_pre':list(memory_usage_pre),
        'ram_post':list(memory_usage_post),
        'gpu_mem_pre':gpu_mem_pre,
        'gpu_mem_post':gpu_mem_post,
        'gpu_load':gpu_load
    }
    return result
