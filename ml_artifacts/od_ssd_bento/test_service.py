from PIL import Image
import requests
import numpy as np
import json
import time,sys
import cv2

imgfile = 'dog.jpg'
port = 4000
request_url = 'edusense-compute-4.andrew.cmu.edu'

if len(sys.argv)>1:
    port = int(sys.argv[1])
print("Testing service on port:", port)
img = cv2.imread(imgfile)
numpy_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

st = time.time()
response = requests.post(
    "http://{request_url}:{port}/detect",
    headers={"content-type": "application/json"},
    data=json.dumps(numpy_img.tolist()))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)

