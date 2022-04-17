import torch
import torchaudio
import bentoml

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
tacotron2 = bundle.get_tacotron2()
tacotron2.eval()
vocoder = bundle.get_vocoder()
vocoder.eval()


processor = bundle.get_text_processor()

# save models to bento
bentoml.pytorch.save('tts_processor',processor)
bentoml.pytorch.save('tts_model',tacotron2)
bentoml.pytorch.save('tts_decoder',vocoder)

# test saved models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts_proc = bentoml.pytorch.load('tts_processor')
tts_model = bentoml.pytorch.load('tts_model')
tts_model.to(DEVICE)
tts_decoder = bentoml.pytorch.load('tts_decoder')
tts_decoder.to(DEVICE)

text = 'Si'
processed, lengths = tts_proc(text)
processed = processed.to(DEVICE)
lengths = lengths.to(DEVICE)
spec, spec_lengths, _ = tts_model.infer(processed, lengths)
waveforms, lengths = tts_decoder(spec, spec_lengths)
np_waveform = waveforms[0:1].squeeze().detach().numpy()
sampling_rate = tts_decoder.sample_rate
print("finished")
pass





