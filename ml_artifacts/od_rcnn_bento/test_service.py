from PIL import Image
import requests
import numpy as np
import json
import time
import cv2

imgfile = 'dog.jpg'
img = cv2.imread(imgfile)
numpy_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

st = time.time()
response = requests.post(
    "http://0.0.0.0:4000/detect",
    headers={"content-type": "application/json"},
    data=json.dumps(numpy_img.tolist()))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)

