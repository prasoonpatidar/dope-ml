import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from dope_ml_model import *

model_path = "trained_model.pt"
model = LinearModel()


def load_model():

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

def get_resource_prediction(app_name, model_name, no_of_requests, cpu_plan, cpu_mem_plan, gpu_plan):
    app_feat = get_app_feature(app_name)
    model_feat = get_model_feature(model_name)
    input_features = []
    input_features.append(app_feat)
    input_features.append(model_feat)
    input_features.append(no_of_requests)
    input_features.append(cpu_plan)
    input_features.append(cpu_mem_plan)
    input_features.append(gpu_plan)

    input_features = np.reshape(1,19)

    with torch.no_grad():
        predicted_resources = model(input_features)

    # List of [cpu_usage, gpu_wrk_util, max_mem, max_gpu_wrk_mem]
    return predicted_resources