import requests
import json
import time
import sys

port = 3000
if len(sys.argv)>1:
    port = int(sys.argv[1])
print("Testing service on port:", port)
request_url = 'edusense-compute-4.andrew.cmu.edu'
request_url = '0.0.0.0'

request_payload = {
    'input':"Tooth",
}
st = time.time()
response = requests.post(
    f"http://{request_url}:{port}/convert",
    headers={"content-type": 'application/json'},
    data=json.dumps(request_payload))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)

#conda config --set auto_activate_base false