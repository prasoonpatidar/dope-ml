from fastapi import FastAPI
from .config import settings
from .models import *
import requests
import json
import time

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference/{model}")
async def read_item(model: str):
    request_payload = {
        'input': "Tooth",
    }
    st = time.time()
    response = requests.post(
        f"http://{settings.gpuClusterConfigUrl}:{settings.gpuClusterConfigPort}/url",
        headers={"content-type": 'application/json'},
        data=json.dumps(request_payload))
    total_time = time.time() - st
    response_dict = json.loads(response.text)
    print(total_time)

    return {"item_id": model}


@app.post("/inference")
async def post_item(data: ClientData):
    # Check which backend instance to schedule the device on.

    if not data.model or not data.resource or not data.payload:
        return {"success": "false"}

    gpu_present_flag = False
    bento_instance = settings.B1

    if data.resource["gpu"] != 0:
        gpu_present_flag = True

    if data.resource["cpu"] <= 50:
        if gpu_present_flag:
            bento_instance = settings.B3
        else:
            bento_instance = settings.B1
    elif data.resource["cpu"] > 50:
        if gpu_present_flag:
            bento_instance = settings.B4
        else:
            bento_instance = settings.B2

    if data.model == "bert":
        bento_payload = BertModel(input=data.payload["input"],
                                  masked_index=data.payload["masked_index"],
                                  segments_ids=data.payload["segments_ids"])
        st = time.time()
        response = requests.post(
            f"http://{bento_instance['url']}:{bento_instance['bert']['port']}/predict",
            headers={"content-type": 'application/json'},
            data=json.dumps(bento_payload.dict()))
        total_time = time.time() - st
        response_dict = json.loads(response.text)
        print(total_time, str(response_dict))
        # return {"success": "true"}
        return response_dict
    elif data.model == "ssd":
        ssd_payload = SSDModel(data=data.payload)
        st = time.time()
        response = requests.post(
            f"http://{settings.B1['url']}:{settings.B1['bert']['port']}/predict",
            headers={"content-type": 'application/json'},
            data=json.dumps(ssd_payload.dict()))
        total_time = time.time() - st
        response_dict = json.loads(response.text)
        print(total_time, str(response_dict))
        # return {"success": "true"}
        return response_dict
    elif data.model == "resnet":
        resnet_payload = ResNetModel(data=data.payload)
        st = time.time()
        response = requests.post(
            f"http://{settings.B1['url']}:{settings.B1['bert']['port']}/predict",
            headers={"content-type": 'application/json'},
            data=json.dumps(resnet_payload.dict()))
        total_time = time.time() - st
        response_dict = json.loads(response.text)
        print(total_time, str(response_dict))
        return response_dict
    elif data.model == "rnn-t":  # STT
        rnnt_payload = STTModel(np_waveform=data.payload["np_waveform"],
                                sampling_rate=data.payload["sampling_rate"])
        st = time.time()
        response = requests.post(
            f"http://{settings.B1['url']}:{settings.B1['bert']['port']}/predict",
            headers={"content-type": 'application/json'},
            data=json.dumps(rnnt_payload.dict()))
        total_time = time.time() - st
        response_dict = json.loads(response.text)
        print(total_time, str(response_dict))
        return response_dict

    return {"success": "false"}
