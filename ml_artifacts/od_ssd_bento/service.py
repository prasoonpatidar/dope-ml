'''Main service file for object detection model'''
import numpy as np
from PIL import Image as PILImage
import bentoml
from bentoml.io import Image, NumpyNdarray, JSON
import torch, torch.nn
from torchvision import transforms
import time

# Load the model
od_runner = bentoml.pytorch.load_runner('object_detection_ssd:latest')
model = bentoml.pytorch.load('object_detection_ssd:latest')

# create new service
od_svc = bentoml.Service('object_detection',runners=[od_runner])

# preprocess image
def od_preprocess(image):
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    return image

# main api for detection
@od_svc.api(
    input=NumpyNdarray(),
    output= JSON(),
)
def detect(np_input_image):
    start_time = time.time()
    input_tensor = od_preprocess(np.uint8(np_input_image))

    # move the input and model to GPU for speed if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",DEVICE)
    input_tensor = input_tensor.to(DEVICE)
    od_model = model.to(DEVICE)

    # with torch.no_grad():
    output = od_model(input_tensor)[0]
    result = {
        'boxes':output['boxes'].cpu().detach().numpy().tolist(),
        'labels':output['labels'].cpu().detach().numpy().tolist(),
        'scores':output['scores'].cpu().detach().numpy().tolist(),
        'time':time.time()-start_time
    }
    return result
