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
    if data.model == "bert":
        bentoPayload = data.payload
        # bento_Payload = BertModel()
        # bento_Payload.input = data.payload["input"]
        # bento_Payload.masked_index = data.payload["masked_index"]
        # bento_Payload.segments_ids = data.payload["segments_ids"]

        st = time.time()
        response = requests.post(
            f"http://{settings.gpuClusterConfigUrl}:{settings.gpuClusterBertPort}/predict",
            headers={"content-type": 'application/json'},
            data=json.dumps(bentoPayload))
        total_time = time.time() - st
        response_dict = json.loads(response.text)
        print(total_time, str(response_dict))
        return {"success": "true"}
        return response_dict

    return {"success": "false"}
