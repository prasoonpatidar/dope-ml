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

tts_payloads = [
    {
        'input':"Tooth",
    },
    {
        'input':"Tooth is an example",
    },
    {
        'input':"Why only Tooth is an example, Teeth can also be an example",
    },
    {
        'input':"Why only Tooth is an example, Teeth can also be an example. Teeth can also be an example.",
    },
    {
        'input':"Why only Tooth is an example, Teeth can also be an example. Teeth can also be an example. Teeth can also be an example. Some more",
    },
]

img_payloads = [
    np.array(Image.open('img_xs.jpg')).tolist(),
    np.array(Image.open('img_s.jpg')).tolist(),
    np.array(Image.open('img_m.jpg')).tolist(),
    np.array(Image.open('img_l.jpg')).tolist(),
    np.array(Image.open('img_xl.jpg')).tolist()
]

bert_payloads = [
    {
        'input':"Who was Jim Henson ? Jim xx was a puppeteer Hellow this is a testing string",
        'masked_index':8,
        'segments_ids' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    },
    {
        'input':"Who was Jim Henson ? Jim xx was a puppeteer Who knows Jim? If there's Jim, Where is Pam? How can this string be made longer than it is now? What will it affect? Let's see",
        'masked_index':6,
        'segments_ids' : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    },
    {
        'input':"Who was Jim Henson ? Jim xx was a puppeteer",
        'masked_index':6,
        'segments_ids' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    },
    {
        'input':"Who was Jim Henson ? Jim xx was a puppeteer If there's Jim, Where is Pam? How can this",
        'masked_index':6,
        'segments_ids' : [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    },
    {
        'input':"Who was Jim ",
        'masked_index':3,
        'segments_ids' : [0, 0, 1, 1]
    }
]


np_waveform1, sampling_rate1 = sf.read('audio_1.wav')
np_waveform2, sampling_rate2 = sf.read('audio_2.wav')
np_waveform3, sampling_rate3 = sf.read('audio_3.wav')
np_waveform4, sampling_rate4 = sf.read('audio_4.wav')
np_waveform5, sampling_rate5 = sf.read('audio_5.wav')

audio_payloads = [
    {
        'np_waveform':np_waveform1.tolist(),
        'sampling_rate':sampling_rate1
    },
    {
        'np_waveform':np_waveform2.tolist(),
        'sampling_rate':sampling_rate2
    },
    {
        'np_waveform':np_waveform3.tolist(),
        'sampling_rate':sampling_rate3
    },
    {
        'np_waveform':np_waveform4.tolist(),
        'sampling_rate':sampling_rate4
    },
    {
        'np_waveform':np_waveform5.tolist(),
        'sampling_rate':sampling_rate5
    }
]


model_payload_map = {
    "bert": bert_payloads,
    "resnet": img_payloads,
    "ssd": img_payloads,
    "stt": audio_payloads,
    "tts": tts_payloads
}
