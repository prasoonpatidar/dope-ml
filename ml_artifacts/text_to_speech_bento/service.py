import bentoml
import torch
import torchaudio
from bentoml.io import JSON
import time

from queue import Queue
import psutil
from GPUStatMonitor import GPUStatMonitor, get_all_queue_result
import GPUtil
import resource

#setup stat monitor
gpus = GPUtil.getGPUs()
gpu_stat_queue = Queue()


# get model runners
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
tts_processor = processor = bundle.get_text_processor()
# tts_processor = bentoml.pytorch.load('tts_processor:latest')
# tts_processor_runner = bentoml.pytorch.load_runner('tts_processor:latest')

tts_model = bentoml.pytorch.load('tts_model:latest')
tts_model_runner = bentoml.pytorch.load_runner('tts_model:latest')

tts_decoder = bentoml.pytorch.load('tts_decoder:latest')
tts_decoder_runner = bentoml.pytorch.load_runner('tts_decoder:latest')

# tts_svc = bentoml.Service('tts_service', runners=[tts_processor_runner, tts_model_runner, tts_decoder_runner])
tts_svc = bentoml.Service('tts_service', runners=[tts_model_runner, tts_decoder_runner])

@tts_svc.api(input=JSON(), output=JSON())
def convert(request_payload):
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
    parent_start_rusage = resource.getrusage(resource.RUSAGE_SELF)
    children_start_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
    start_time = time.time()
    text_input = request_payload['input']
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", DEVICE)
    tts_model.to(DEVICE)
    tts_decoder.to(DEVICE)
    processed, lengths = tts_processor(text_input)
    processed = processed.to(DEVICE)
    lengths = lengths.to(DEVICE)
    spec, spec_lengths, _ = tts_model.infer(processed, lengths)
    waveforms, lengths = tts_decoder(spec, spec_lengths)
    np_waveform = waveforms.cpu()[0:1].squeeze().detach().numpy()
    sampling_rate = tts_decoder.sample_rate

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

    result =  {
        'audio_out':np_waveform.tolist(),
        'sampling_rate':sampling_rate,
        'time': inf_time,
        'cpu_util': total_cpu_utilization,
        'cpu_times': cpu_times,
        'ram_pre': list(memory_usage_pre),
        'ram_post': list(memory_usage_post),
        'res_cpu_util': cpu_util,
        'res_max_memory': max_mem,
        'gpu_mem_pre': gpu_mem_pre,
        'gpu_mem_post': gpu_mem_post,
        'gpu_load': gpu_load
    }

    return result





