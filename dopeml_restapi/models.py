from typing import Optional
from pydantic import BaseModel
import numpy as np


class BertModel(BaseModel):
    input: str
    masked_index: int
    segments_ids: list


class SSDModel(BaseModel):
    data: list


class ResNetModel(BaseModel):
    data: list
    # data: np.array


class STTModel(BaseModel):
    np_waveform: list
    sampling_rate: float


class TTSModel(BaseModel):
    input: str


class ClientData(BaseModel):
    model: str
    resource: str
    payload: dict
