import requests
import json
import time
import soundfile as sf
from requests_toolbelt.multipart import MultipartEncoder
speech_file = 'output_wavernn.wav'
np_waveform, sampling_rate = sf.read(speech_file)

# m = MultipartEncoder(
#     fields={
#         'np_waveform':np_waveform,
#         'sampling_rate':{'sr':sampling_rate}
#     }
# )
request_payload = {
    'np_waveform':np_waveform.tolist(),
    'sampling_rate':sampling_rate
}
st = time.time()
response = requests.post(
    "http://0.0.0.0:4040/convert",
    headers={"content-type": 'application/json'},
    data=json.dumps(request_payload))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)

