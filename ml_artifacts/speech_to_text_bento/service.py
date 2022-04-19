import time
import numpy as np
import torch
import torchaudio
import bentoml
from bentoml.io import JSON, NumpyNdarray, Multipart

from queue import Queue
import psutil
from GPUStatMonitor import GPUStatMonitor, get_all_queue_result
import GPUtil
import resource

#setup stat monitor
gpus = GPUtil.getGPUs()
gpu_stat_queue = Queue()


model = bentoml.pytorch.load('stt_model:latest')
stt_model_runner = bentoml.pytorch.load_runner('stt_model:latest')
stt_decoder = bentoml.pytorch.load('stt_decoder:latest')
stt_decoder_runner = bentoml.pytorch.load_runner('stt_decoder:latest')
stt_svc = bentoml.Service('stt_service',runners=[stt_model_runner, stt_decoder_runner])

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
bundle_sampling_rate = bundle.sample_rate
bundle_labels = bundle.get_labels()

#get input and output specs
# input_spec = Multipart(
#     np_waveform=NumpyNdarray(),
#     sampling_rate=JSON()
# )

@stt_svc.api(input=JSON(), output=JSON())
def convert(request_json):
    # stats monitoring code
    psutil.cpu_percent(interval=0.1)
    memory_usage_pre = psutil.virtual_memory()
    if len(gpus) > 0:
        gpu_mem_pre = [gpu_device.memoryUtil for gpu_device in gpus]
        gpu_m = GPUStatMonitor(gpu_stat_queue)
        gpu_m.start()
    else:
        gpu_mem_pre = []

    #actual runtime
    parent_start_rusage = resource.getrusage(resource.RUSAGE_SELF)
    children_start_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    start_time = time.time()

    # get input
    np_waveform = np.array(request_json['np_waveform'])
    sampling_rate = request_json['sampling_rate']

    waveform = torch.from_numpy(np_waveform).float().unsqueeze(0)
    if sampling_rate != bundle_sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, bundle_sampling_rate)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", DEVICE)
    waveform = waveform.to(DEVICE)
    stt_model = model.to(DEVICE)
    emission, _ = stt_model(waveform)

    transcript = stt_decoder(emission.cpu()[0])

    # stats monitoring code
    inf_time = time.time() - start_time
    cpu_times = psutil.cpu_percent(interval=inf_time, percpu=True)
    total_cpu_utilization = psutil.cpu_percent(interval=None)

    # new stats method
    children_end_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    parent_end_rusage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_util = (children_end_rusage.ru_utime - children_start_rusage.ru_utime + parent_end_rusage.ru_utime -
                parent_start_rusage.ru_utime) / inf_time * 100
    max_mem = (children_end_rusage.ru_maxrss + parent_end_rusage.ru_maxrss) / (1024 * 1024 * 1024)

    if len(gpus) > 0:
        gpu_mem_post = [gpu_device.memoryUtil for gpu_device in gpus]
        gpu_m.stop()
        gpu_load = get_all_queue_result(gpu_stat_queue)
    else:
        gpu_mem_post, gpu_load = [], []
    memory_usage_post = psutil.virtual_memory()

    result = {
        'transcript':transcript,
        'time':inf_time,
        'cpu_util':total_cpu_utilization,
        'cpu_times':cpu_times,
        'ram_pre':list(memory_usage_pre),
        'ram_post':list(memory_usage_post),
        'res_cpu_util': cpu_util,
        'res_max_memory': max_mem,
        'gpu_mem_pre':gpu_mem_pre,
        'gpu_mem_post':gpu_mem_post,
        'gpu_load':gpu_load
    }
    return result


