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
    np_waveform = np.array(request_json['np_waveform'])
    sampling_rate = request_json['sampling_rate']
    start_time =time.time()
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
    total_cpu_utilization = psutil.cpu_percent(interval=None)
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
        'cpu_times':psutil.cpu_percent(interval=inf_time, percpu=True),
        'ram_pre':list(memory_usage_pre),
        'ram_post':list(memory_usage_post),
        'gpu_mem_pre':gpu_mem_pre,
        'gpu_mem_post':gpu_mem_post,
        'gpu_load':gpu_load
    }
    return result


