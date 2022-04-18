'''
Training Bert with Masked Language Modelling
'''
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import bentoml

# load models
class Tokenizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, emission: torch.Tensor) -> str:

        return ""

    def convert_tokens_to_ids(self,text):
        return self.tokenizer.convert_tokens_to_ids(text)

    def convert_ids_to_tokens(self, predicted_arr):
        return self.tokenizer.convert_ids_to_tokens(predicted_arr)

    def tokenize(self, tokenize_input):
        return self.tokenizer.tokenize(tokenize_input)

tokenizer = Tokenizer()
masked_lm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
masked_lm_model.eval()

bentoml.pytorch.save('bert_tokenizer', tokenizer)
bentoml.pytorch.save('bert_lm_model',masked_lm_model)

# test saved models
bert_tokenizer  = bentoml.pytorch.load('bert_tokenizer:latest')
bert_lm_model = bentoml.pytorch.load('bert_lm_model:latest')

test_input = "Who was Jim Henson ? Jim Henson was a puppeteer"
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
masked_index = 6

tokenized_text = bert_tokenizer.tokenize(test_input)
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

#call model
predictions = bert_lm_model(tokens_tensor, segments_tensors)
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = bert_tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)



