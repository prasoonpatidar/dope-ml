import profile
import requests
import json
import time
import pandas as pd
import random
from sklearn.utils import shuffle


from backends_meta import *
# 'time': inf_time,
#         'cpu_util': total_cpu_utilization,
#         'cpu_times': psutil.cpu_percent(interval=inf_time, percpu=True),
#         'ram_pre': list(memory_usage_pre),
#         'ram_post': list(memory_usage_post),
#         'gpu_mem_pre': gpu_mem_pre,
#         'gpu_mem_post': gpu_mem_post,
#         'gpu_load': gpu_load
profiled_df = pd.DataFrame(columns=['app_name', 'model_name', 'model_type', 'model_layers', 'model_hiddens', 'inf_time', 'input_dimens', 'cpu_util', 'gpu_util', 'cpu_mem', 'gpu_mem'])

def do_inference(app, model, backend):

    global profiled_df
    port = model_backend_map[model][backend]
    endpoint = model_endpoint_map[model]
    payload = model_payload_map[model]

    payload_json = json.dumps(payload)    
    response = requests.post(
        f"http://{request_url}:{port}/{endpoint}",
        headers={"content-type": 'application/json'},
        data=payload_json)

    input_length = len(payload_json)

    response_dict = json.loads(response.text)
    # print(response_dict)

    gpu_mem = 0 if response_dict['gpu_mem_post'] == [] else response_dict['gpu_mem_post'][0]
    gpu_util = 0 if ['gpu_load'] == [] else ['gpu_load'][0][0]
    
    got_values = [app, 
                    model, 
                    model_characteristics[model]['type'], 
                    model_characteristics[model]['no_of_layers'], 
                    model_characteristics[model]['hidden_size'], 
                    response_dict['time'],
                    input_length,
                    response_dict['cpu_util'],
                    # response_dict['gpu_util'],
                    gpu_util,
                    # response_dict['cpu_mem'],
                    response_dict['ram_post'][0],
                    gpu_mem]

    profiled_df = pd.DataFrame(np.insert(profiled_df.values, 0, values=got_values, axis=0))


def main():
    global profiled_df
    for model in available_models.keys():
        app_name = available_models[model]
        for i in range(0,1):
            be = list(model_backend_map[model].keys())
            idx = random.randint(0, len(be)-1)
            selected_be = be[idx]
            do_inference(app_name, model, selected_be)
    
    profiled_df = shuffle(profiled_df)
    profiled_df.to_csv("models_profile.csv")



if __name__ == "__main__":
    main()