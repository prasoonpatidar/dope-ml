import bentoml
import torch
import torchaudio
from bentoml.io import JSON

# get device for torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# get model runners
# bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
# tts_processor = processor = bundle.get_text_processor()
tts_processor = bentoml.pytorch.load('tts_processor:latest')
tts_processor_runner = bentoml.pytorch.load_runner('tts_processor:latest')

tts_model = bentoml.pytorch.load('tts_model:latest')
tts_model_runner = bentoml.pytorch.load_runner('tts_model:latest')

tts_decoder = bentoml.pytorch.load('tts_decoder:latest')
tts_decoder_runner = bentoml.pytorch.load_runner('tts_decoder:latest')

tts_svc = bentoml.Service('tts_service', runners=[tts_processor_runner, tts_model_runner, tts_decoder_runner])
# tts_svc = bentoml.Service('tts_service', runners=[tts_model_runner, tts_decoder_runner])

@tts_svc.api(input=JSON(), output=JSON())
def convert(request_payload):
    text_input = request_payload['input']
    tts_model.to(device)
    tts_decoder.to(device)
    processed, lengths = tts_processor(text_input)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tts_model.infer(processed, lengths)
    waveforms, lengths = tts_decoder(spec, spec_lengths)
    np_waveform = waveforms[0:1].squeeze().detach().numpy()
    sampling_rate = tts_decoder.sample_rate
    return {
        'audio_out':np_waveform.tolist(),
        'sampling_rate':sampling_rate
    }





