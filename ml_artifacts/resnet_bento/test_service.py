from PIL import Image
import requests
import numpy as np
import json
import time

imgfile = 'dog.jpg'

numpy_img = np.array(Image.open('dog.jpg'))
st = time.time()
response = requests.post(
    "http://0.0.0.0:9000/classify",
    headers={"content-type": "application/json"},
    data=json.dumps(numpy_img.tolist()))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)
