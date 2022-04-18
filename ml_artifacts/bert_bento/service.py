import bentoml
from bentoml.io import JSON
import time
import torch

from queue import Queue
import psutil
from GPUStatMonitor import GPUStatMonitor, get_all_queue_result
import GPUtil

#setup stat monitor
gpus = GPUtil.getGPUs()
gpu_stat_queue = Queue()

# load models
bert_tokenizer  = bentoml.pytorch.load('bert_tokenizer:latest')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = bentoml.pytorch.load('bert_lm_model:latest')

bert_tokenizer_runner  = bentoml.pytorch.load_runner('bert_tokenizer:latest')
bert_lm_model_runner = bentoml.pytorch.load_runner('bert_lm_model:latest')

# create bert service
bert_svc = bentoml.Service('bert_service', runners=[bert_tokenizer_runner, bert_lm_model_runner])


# api for service call
@bert_svc.api(input=JSON(), output=JSON())
def predict(request_payload):
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
    text_input = request_payload['input']
    masked_index = request_payload['masked_index']
    segments_ids = request_payload['segments_ids']
    start_time = time.time()

    # get tokenized input
    tokenized_text = bert_tokenizer.tokenize(text_input)
    # print(tokenized_text)
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    # print(tokens_tensor)
    segments_tensors = torch.tensor([segments_ids])
    # print(segments_tensors)

    # check device configs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)
    bert_lm_model = bert_model.to(DEVICE)
    tokens_tensor = tokens_tensor.to(DEVICE)
    segments_tensors = segments_tensors.to(DEVICE)

    # call model
    predictions = bert_lm_model(tokens_tensor, segments_tensors)
    predictions = predictions.cpu()
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = bert_tokenizer.convert_ids_to_tokens([predicted_index])[0]

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

    # return predicted token
    result =  {
        'predicted_token':predicted_token,
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