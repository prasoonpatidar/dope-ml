import numpy as np
from PIL import Image as PILImage
import bentoml
from bentoml.io import Image, NumpyNdarray
import torch, torch.nn
from torchvision import transforms


# load the runner
resnet_runner = bentoml.pytorch.load_runner("resnet:latest")
resnet_model = bentoml.pytorch.load("resnet:latest")
# create a new service
resnet_svc = bentoml.Service("resnet_service",runners=[resnet_runner])


# preprocess incoming image
def resnet_preprocess(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img)


# main service wrapper for deployment
@resnet_svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(np_input_image):
    input_image = PILImage.fromarray(np.uint8(np_input_image))

    input_tensor = resnet_preprocess(input_image)
    print(input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    print(input_batch.shape)
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        resnet_model.cuda()

    # with torch.no_grad():
    output = resnet_model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities.detach().numpy()
