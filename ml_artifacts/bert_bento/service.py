import bentoml
from bentoml.io import JSON
import time
import torch

# load models
bert_tokenizer  = bentoml.pytorch.load('bert_tokenizer:latest')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_lm_model = bentoml.pytorch.load('bert_lm_model:latest')

bert_tokenizer_runner  = bentoml.pytorch.load_runner('bert_tokenizer:latest')
bert_lm_model_runner = bentoml.pytorch.load_runner('bert_lm_model:latest')

# create bert service
bert_svc = bentoml.Service('bert_service', runners=[bert_tokenizer_runner, bert_lm_model_runner])


# api for service call
@bert_svc.api(input=JSON(), output=JSON())
def predict(request_payload):
    text_input = request_payload['input']
    masked_index = request_payload['masked_index']
    segments_ids = request_payload['segments_ids']
    start_time = time.time()

    # get tokenized input
    tokenized_text = bert_tokenizer.tokenize(text_input)
    # print(tokenized_text)
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    # print(tokens_tensor)
    segments_tensors = torch.tensor([segments_ids])
    # print(segments_tensors)

    # call model
    predictions = bert_lm_model(tokens_tensor, segments_tensors)
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = bert_tokenizer.convert_ids_to_tokens([predicted_index])[0]

    # return predicted token
    return {
        'predicted_token':predicted_token,
        'time':time.time()-start_time,
    }