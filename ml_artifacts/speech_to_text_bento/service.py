import time
import numpy as np
import torch
import torchaudio
import bentoml
from bentoml.io import JSON, NumpyNdarray, Multipart

stt_model = bentoml.pytorch.load('stt_model:latest')
stt_model_runner = bentoml.pytorch.load_runner('stt_model:latest')
stt_decoder = bentoml.pytorch.load('stt_decoder:latest')
stt_decoder_runner = bentoml.pytorch.load_runner('stt_decoder:latest')
stt_svc = bentoml.Service('stt_service',runners=[stt_model_runner, stt_decoder_runner])

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
bundle_sampling_rate = bundle.sample_rate
bundle_labels = bundle.get_labels()

#get input and output specs
# input_spec = Multipart(
#     np_waveform=NumpyNdarray(),
#     sampling_rate=JSON()
# )

@stt_svc.api(input=JSON(), output=JSON())
def convert(request_json):
    np_waveform = np.array(request_json['np_waveform'])
    sampling_rate = request_json['sampling_rate']
    start_time =time.time()
    waveform = torch.from_numpy(np_waveform).float().unsqueeze(0)
    if sampling_rate != bundle_sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sampling_rate, bundle_sampling_rate)

    emission, _ = stt_model(waveform)
    transcript = stt_decoder(emission[0])

    return {
        'transcript':transcript,
        'time':time.time()-start_time
    }


