import numpy as np
from PIL import Image as PILImage
import bentoml
from bentoml.io import Image, NumpyNdarray, JSON
import torch, torch.nn
from torchvision import transforms
import time
import json
from queue import Queue
import psutil
from GPUStatMonitor import GPUStatMonitor, get_all_queue_result
import GPUtil

#setup stat monitor
gpus = GPUtil.getGPUs()
gpu_stat_queue = Queue()
gpu_m=GPUStatMonitor(gpu_stat_queue)

# load the runner
resnet_runner = bentoml.pytorch.load_runner("resnet:latest")
resnet_model = bentoml.pytorch.load("resnet:latest")
# create a new service
resnet_svc = bentoml.Service("resnet_service",runners=[resnet_runner])


# preprocess incoming image
def resnet_preprocess(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img)


# main service wrapper for deployment
@resnet_svc.api(
    input=NumpyNdarray(),
    output= JSON(),
)
def classify(np_input_image):
    psutil.cpu_percent(interval=0.1)
    memory_usage_pre = psutil.virtual_memory()
    if len(gpus) > 0:
        gpu_mem_pre = [gpu_device.memoryUtil for gpu_device in gpus]
        gpu_m.start()
    else:
        gpu_mem_pre = []
    input_image = PILImage.fromarray(np.uint8(np_input_image))
    start_time = time.time()
    input_tensor = resnet_preprocess(input_image)
    print(input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    print(input_batch.shape)
    # move the input and model to GPU for speed if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)
    input_batch = input_batch.to(DEVICE)
    resnet_model_device = resnet_model.to(DEVICE)

    # with torch.no_grad():
    output = resnet_model_device(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu()
    inf_time = time.time() - start_time
    total_cpu_utilization = psutil.cpu_percent(interval=None)
    if len(gpus) > 0:
        gpu_mem_post = [gpu_device.memoryUtil for gpu_device in gpus]
        gpu_m.stop()
        gpu_load = get_all_queue_result(gpu_stat_queue)
    else:
        gpu_mem_post,gpu_load = [], []
    memory_usage_post = psutil.virtual_memory()
    result = {
        'probabilities':probabilities.detach().numpy().tolist(),
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
