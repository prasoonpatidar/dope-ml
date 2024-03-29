from PIL import Image
import requests
import numpy as np
import json
import time
import sys

imgfile = 'dog.jpg'
port = 3000
request_url = 'edusense-compute-4.andrew.cmu.edu'
request_url = '0.0.0.0'

if len(sys.argv)>1:
    port = int(sys.argv[1])
print("Testing service on port:", port)
numpy_img = np.array(Image.open('dog.jpg'))
i=0
st = time.time()
response = requests.post(
        f"http://{request_url}:{port}/classify",
    headers={"content-type": "application/json"},
    data=json.dumps(numpy_img.tolist()))
total_time = time.time()-st
response_dict = json.loads(response.text)
response_dict['response_time'] = total_time
print(i, total_time,response_dict['time'])

