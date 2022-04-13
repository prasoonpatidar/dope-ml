import torch
import torchaudio
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# manual symbol transformation
# symbols = '_-!\'(),.:;? abcdefghijklmnopqrstuvwxyz'
# look_up = {s: i for i, s in enumerate(symbols)}
# symbols = set(symbols)
#
# def text_to_sequence(text):
#   text = text.lower()
#   return [look_up[s] for s in text if s in symbols]
#
# text = "Hello world! Text to speech!"
# print(text_to_sequence(text))

# Singleton Preprocessing
# processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()
#
# text = "Hello world! Text to speech!"
# processed, lengths = processor(text)
#
# print(processed)
# print(lengths)
# print([processor.tokens[i] for i in processed[0, :lengths[0]]])


# phenome based encoding
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()

text = "Privacy Matters"
with torch.inference_mode():
  processed, lengths = processor(text)

print(processed)
print(lengths)
print([processor.tokens[i] for i in processed[0, :lengths[0]]])

# spectrogram generation using tacotron2

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)

text = "Stay Hungry, Stay Foolish"

with torch.inference_mode():
  processed, lengths = processor(text)
  processed = processed.to(device)
  lengths = lengths.to(device)
  spec, _, _ = tacotron2.infer(processed, lengths)


plt.imshow(spec[0].cpu().detach())
plt.show()

# Generating waveform from spectrogram
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH

processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

text = "Stay Hungry, Stay Foolish"

with torch.inference_mode():
  processed, lengths = processor(text)
  processed = processed.to(device)
  lengths = lengths.to(device)
  spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
  waveforms, lengths = vocoder(spec, spec_lengths)

torchaudio.save("output_wavernn.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
# IPython.display.display(IPython.display.Audio("output_wavernn.wav"))
pass