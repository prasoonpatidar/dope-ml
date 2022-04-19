import soundfile as sf
from PIL import Image
import numpy as np


available_models = { "resnet" : "Image classification", 
                    "stt" : "Speech-to-text", 
                    "ssd" : "Object detection", 
                    "tts" : "Text-to-speech", 
                    "bert" : "Natural Language Processing" }

# available_models = ["resnet", "stt", "ssd", "tts", "bert"]

backend_configs = {
    "2C0G2M": [200, 0, 2],
    "2C0G3M": [200, 0, 3],
    "2C0G4M": [200, 0, 4],
    "4C0G4P5M": [200, 0, 4.5],
    "4C0G4M": [400, 0, 4],
    "4C1G6M": [400, 100, 6]
}


model_backend_map = {
    "resnet": {"2C0G2M": 4001, 
            "4C0G4M": 5001, 
            "2C0G3M": 6001,
            "4C1G6M": 7001 },
    "ssd": {"2C0G2M": 4002, 
            "4C0G4M": 5002, 
            "2C0G3M": 6002,
            "4C1G6M": 7002 },
    "stt": {"2C0G3M": 4003, 
            "4C0G4M": 5003, 
            "2C0G4.5M": 6003,
            "4C1G6M": 7003 },
    "tts": {"2C0G2M": 4004, 
            "4C0G4M": 5004, 
            "2C0G4M": 6004,
            "4C1G6M": 7004 },
    "bert": {"2C0G3M": 4005, 
            "4C0G4M": 5005, 
            "2C0G4M": 6005,
            "4C1G6M": 7005 }
}


model_endpoint_map = {
    "resnet": "classify",
    "ssd": "detect",
    "tts": "convert",
    "stt": "convert",
    "bert": "predict"
}


model_types = ["cnn", "mlp", "unet", "rnn", "transformer", "gen"]


model_characteristics = {
    "resnet": {"type": "cnn", "no_of_layers": 18, "hidden_size": 0}, 
    "ssd": {"type": "cnn", "no_of_layers": 50, "hidden_size": 0}, 
    "stt": {"type": "transformer", "no_of_layers": 19, "hidden_size": 512}, 
    "tts": {"type": "transformer", "no_of_layers": 19, "hidden_size": 512}, 
    "bert": {"type": "transformer", "no_of_layers": 12, "hidden_size": 768} 
}


request_url = 'edusense-compute-4.andrew.cmu.edu'
# request_url = '0.0.0.0'


speech_file = 'output_wavernn.wav'
np_waveform, sampling_rate = sf.read(speech_file)
img_file = 'dog.jpg'

model_payload_map = {
    "bert": {
        'input':"Who was Jim Henson ? Jim xx was a puppeteer",
        'masked_index':6,
        'segments_ids' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    },
    "resnet": np.array(Image.open(img_file)).tolist(),
    "ssd": np.array(Image.open(img_file)).tolist(),
    "stt": {
        'np_waveform':np_waveform.tolist(),
        'sampling_rate':sampling_rate
    },
    "tts": {
    'input':"Tooth",
    }
}
