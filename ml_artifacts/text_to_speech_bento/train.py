import torch
import torchaudio
import bentoml

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
device='cpu'
# tacotron2 = bundle.get_tacotron2()
# tacotron2.eval()
# vocoder = bundle.get_vocoder()
# vocoder.eval()

#wrap processor to nn.module
class TTSModel(torch.nn.Module):
    def __init__(self,bundle, device):
        super().__init__()
        self.processor = bundle.get_text_processor()
        self.vocoder = bundle.get_vocoder()
        self.device = device
        self.tacotron2 = bundle.get_tacotron2()

    def forward(self):
        return ""

    def get_waveform(self,text):
        processed, lengths = self.processor(text)
        processed = processed.to(self.device)
        lengths = lengths.to(self.device)
        spec, spec_lengths, _ = self.tacotron2.infer(processed, lengths)
        waveforms, lengths = self.vocoder(spec, spec_lengths)
        return waveforms, lengths

    def eval(self):
        self.vocoder.eval()
        self.tacotron2.eval()
        super().eval()

    def to(self,device):
        self.vocoder.to(device)
        self.tacotron2.to(device)
        super().to(device)

    def sample_rate(self):
        return self.vocoder.sample_rate

tts_model = TTSModel(bundle, device)
tts_model.eval()
bentoml.pytorch.save('tts_model2',tts_model)


saved_model = bentoml.pytorch.load('tts_model:latest')
saved_model.to(device)
text = 'Si'
waveforms, lengths = saved_model.get_waveform(text)
np_waveform = waveforms[0:1].squeeze().detach().numpy()
sampling_rate = saved_model.sample_rate()
print("finished")
pass

# processor = Processor(bundle)

# save models to bento
# # bentoml.pytorch.save('tts_processor',processor)
# bentoml.pytorch.save('tts_model',tacotron2)
# # bentoml.pytorch.save('tts_decoder',vocoder)
#
# # test saved models
#
# tts_proc = bentoml.pytorch.load('tts_processor')
# tts_model = bentoml.pytorch.load('tts_model')
# tts_model.to(device)
# tts_decoder = bentoml.pytorch.load('tts_decoder')
# tts_decoder.to(device)
#
# text = 'Stay Hungry, Stay Foolish'
# processed, lengths = tts_proc(text)
# processed = processed.to(device)
# lengths = lengths.to(device)
# spec, spec_lengths, _ = tts_model.infer(processed, lengths)
# waveforms, lengths = tts_decoder(spec, spec_lengths)
# np_waveform = waveforms[0:1].squeeze().detach().numpy()
# sampling_rate = tts_decoder.sample_rate
# print("finished")
# pass





