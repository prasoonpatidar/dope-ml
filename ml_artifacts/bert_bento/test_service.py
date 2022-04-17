import requests
import json
import time

request_payload = {
    'input':"Who was Jim Henson ? Jim xx was a puppeteer",
    'masked_index':6,
    'segments_ids' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
st = time.time()
response = requests.post(
    "http://0.0.0.0:4040/predict",
    headers={"content-type": 'application/json'},
    data=json.dumps(request_payload))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time,str(response_dict))

'''
tensor([[ 2040,  2001,  3958, 27227,  1029,  3958,   103, 25426,  2001,  1037,
         13997, 11510]])
tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

tensor([[ 2040,  2001,  3958, 27227,  1029,  3958,   103,  2001,  1037, 13997,
         11510]])
tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

'''