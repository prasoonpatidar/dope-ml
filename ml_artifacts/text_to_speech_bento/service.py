import bentoml
import torch
import torchaudio
from bentoml.io import JSON

# get device for torch
device = "cpu"

# get model runners
tts_model = bentoml.pytorch.load('tts_model:latest')
tts_model_runner = bentoml.pytorch.load_runner('tts_model:latest')

tts_svc = bentoml.Service('tts_service', runners=[tts_model_runner])

@tts_svc.api(input=JSON(), output=JSON())
def convert(request_payload):
    text_input = request_payload['input']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", DEVICE)
    tts_model.to(DEVICE)
    waveforms, lengths = tts_model.get_waveform(text_input, DEVICE)
    np_waveform = waveforms[0:1].squeeze().detach().numpy()
    sampling_rate = tts_model.sample_rate()
    return {
        'audio_out':np_waveform.tolist(),
        'sampling_rate':sampling_rate
    }




