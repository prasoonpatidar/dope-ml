import requests
import json
import time
import soundfile as sf
import sys

port = 6003
request_url = 'edusense-compute-4.andrew.cmu.edu'
# request_url = '0.0.0.0'

if len(sys.argv)>1:
    port = int(sys.argv[1])
print("Testing service on port:", port)
speech_file = 'output_wavernn.wav'
np_waveform, sampling_rate = sf.read(speech_file)
request_payload = {
    'np_waveform':np_waveform.tolist(),
    'sampling_rate':sampling_rate
}
st = time.time()
response = requests.post(
    f"http://{request_url}:{port}/convert",
    headers={"content-type": 'application/json'},
    data=json.dumps(request_payload))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)

