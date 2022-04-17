import torch
import torchaudio
import bentoml
import soundfile as sf

# get relevant models from torchaudio bundles
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
stt_model = bundle.get_model()
stt_model = stt_model.float()
stt_model.eval()
bentoml.pytorch.save('stt_model',stt_model)
bundle_sampling_rate = bundle.sample_rate
bundle_labels = bundle.get_labels()

# initialize decoder from stt_model
# Decoding steps: Decode extracted labels with greedy algorithms

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
decoder.eval()
bentoml.pytorch.save('stt_decoder',decoder)

# test stt model and decoder
saved_model = bentoml.pytorch.load('stt_model:latest')
saved_decoder = bentoml.pytorch.load('stt_decoder:latest')
test_file = "output_wavernn.wav"
np_waveform, in_sampling_rate = sf.read(test_file)
waveform = torch.from_numpy(np_waveform).float().unsqueeze(0)

#get emission values
if in_sampling_rate != bundle_sampling_rate:
    waveform = torchaudio.functional.resample(waveform, in_sampling_rate, bundle_sampling_rate)

emission, _ = saved_model(waveform)
transcript = saved_decoder(emission[0])
print(transcript)




