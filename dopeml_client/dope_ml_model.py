import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

def get_app_feature(app_name):
    return np.zeros(4)

def get_model_feature(model_name):
    return np.zeros(4)

dropout = 0.3


class LinearModel(nn.Module):
  def __init__(self):
    super(LinearModel, self).__init__()
    self.layers = nn.Sequential(
        nn.Linear(19, 32),
        nn.Dropout(dropout),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 64),
        nn.Dropout(dropout),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 128),
        nn.Dropout(dropout),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Linear(128,64),
        nn.Dropout(dropout),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.Linear(64,32),
        nn.Dropout(dropout),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 16),
        nn.Dropout(dropout),
        nn.BatchNorm1d(16),
        nn.ReLU(inplace=True),
        nn.Linear(16, 8),
        nn.Dropout(dropout),
        nn.BatchNorm1d(8),
        nn.ReLU(inplace=True)
    )
    self.fc5 = nn.Linear(8, 4)
    self.fc6 = nn.Linear(8, 5) # GPU types
  def forward(self, x):
    x = self.layers(x)
    pred_val = F.relu(self.fc5(x))
    # pred_cat = self.fc6(x)
    return pred_val