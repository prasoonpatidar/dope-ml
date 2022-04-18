import numpy as np
import pandas as pd
import os
import sys
import requests


def getPayload(model_type):
    # load paylaod
    print("1")

workload_file = "workload_input.csv"

dope_ml_endpoint = "endpoint/inference"

workload_df = pd.read_csv(workload_file)

for index, row in workload_df.iterrows():
    request_data = {}
    request_data['model'] = row['workload']
    # request_data['resource'] = getResources(torch.tensor(train_x.values, dtype=torch.float32))
    request_data['payload'] = getPayload(row['workload'])
    # print(row['c1'], row['c2'])
