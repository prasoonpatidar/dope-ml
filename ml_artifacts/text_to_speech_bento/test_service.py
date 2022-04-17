import requests
import json
import time

request_payload = {
    'input':"To the",
}
st = time.time()
response = requests.post(
    "http://0.0.0.0:4000/convert",
    headers={"content-type": 'application/json'},
    data=json.dumps(request_payload))
total_time = time.time()-st
response_dict = json.loads(response.text)
print(total_time)

