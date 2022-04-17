from torchvision.models.detection import ssd300_vgg16
import numpy as np
import pickle
import torch
import cv2
import bentoml
import time


# script arguments
labels_file = 'coco.pickle'
min_confidence = 0.5

# build object detection model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssd_model = ssd300_vgg16(pretrained=True, pretrained_backbone=True).to(DEVICE)
ssd_model.eval()
# save object detection model
tag = bentoml.pytorch.save('object_detection_ssd',ssd_model)

# test saved models
# test saved model
start_time = time.time()
saved_od_model = bentoml.pytorch.load("object_detection_ssd:latest")

CLASSES = pickle.loads(open(labels_file, "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the image from disk
image = cv2.imread('dog.jpg')
orig = image.copy()
# convert the image from BGR to RGB channel ordering and change the
# image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))
# add the batch dimension, scale the raw pixel intensities to the
# range [0, 1], and convert the image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)
# send the input to the device and pass the it through the network to
# get the detections and predictions
# image = image.to(DEVICE)

detections = saved_od_model(image)[0]


# loop over the detections
for i in range(0, len(detections["boxes"])):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections["scores"][i]
    # filter out weak detections by ensuring the confidence is
    # greater than the minimum confidence
    if confidence > min_confidence:
        # extract the index of the class label from the detections,
        # then compute the (x, y)-coordinates of the bounding box
        # for the object
        idx = int(detections["labels"][i])
        box = detections["boxes"][i].detach().cpu().numpy()
        (startX, startY, endX, endY) = box.astype("int")
        # display the prediction to our terminal
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))
        # draw the bounding box and label on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY),
            COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(orig, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
# show the output image
cv2.imshow("Output", orig)
cv2.waitKey(0)