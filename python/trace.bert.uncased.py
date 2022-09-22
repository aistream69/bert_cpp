# -*- coding: utf-8 -*-

from transformers import BertModel, BertForMaskedLM, BertTokenizer, BertConfig
import torch

enc = BertTokenizer(
    "./utils/bert-base-uncased/vocab.txt",
    do_lower_case=False,
    do_basic_tokenize=False
)

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

print(len(tokenized_text), tokenized_text)
# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
print(len(tokenized_text), tokenized_text)
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
#segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
segments_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(indexed_tokens)

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig.from_json_file(
    "./utils/bert-base-uncased/config.json"
)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertForMaskedLM.from_pretrained(
    "./utils/bert-base-uncased/pytorch_model.bin",
    config=config
)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors], strict=False)
torch.jit.save(traced_model, "./utils/bert-base-uncased/traced_bert.pt")

