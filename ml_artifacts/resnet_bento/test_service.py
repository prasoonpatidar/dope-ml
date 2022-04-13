from PIL import Image
import requests
import numpy as np
import json

imgfile = 'dog.jpg'

numpy_img = np.array(Image.open('dog.jpg'))
response = requests.post(
    "http://127.0.0.1:3000/classify",
    headers={"content-type": "application/json"},
    data=json.dumps(numpy_img.tolist()))



