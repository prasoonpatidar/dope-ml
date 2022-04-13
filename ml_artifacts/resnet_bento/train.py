import urllib
from PIL import Image
import torch
from torchvision import transforms, models
import bentoml


# get resnet model
resnet = models.resnet18(pretrained=True)

# save bentoml runner
tag = bentoml.pytorch.save("resnet",resnet)
# bentoml.save()
# test saved model
saved_resnet_model = bentoml.pytorch.load("resnet:latest")
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
with torch.no_grad():
    output = saved_resnet_model(input_batch)

print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


#




