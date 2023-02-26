import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch

def process_docs(path):
    a = []
    with open(path, encoding='latin') as f:
        for line in f.readlines():
            line = line[:-1].replace("  ", "")

            if len(line):
                a.append(line)
    return a


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chkpt = 'vblagoje/bart_lfqa'
model = AutoModelForSeq2SeqLM.from_pretrained(chkpt)
tokenizer = AutoTokenizer.from_pretrained(chkpt)
model = model.to(device)
dir_path = 'C:/Users/ksara/OneDrive/Desktop/assign/openfabric-test/dataset/'
books = os.listdir(dir_path)
paths = [dir_path + book for book in books]

array = list(map(process_docs, paths))
context = [line for sub_array in array for line in sub_array]

conditioned_context = "<P> " + " <P> ".join([d for d in context])

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass




############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []

    for text in request.text:
        query_and_docs = "question: {} context: {}".format(text, conditioned_context)

        model_input = tokenizer(query_and_docs, truncation=True, padding=True, return_tensors="pt")

        generated_answers_encoded = model.generate(input_ids=model_input["input_ids"].to(device),
                                                   attention_mask=model_input["attention_mask"].to(device),
                                                   min_length=16,
                                                   max_length=128,
                                                   do_sample=False,
                                                   early_stopping=True,
                                                   num_beams=8,
                                                   temperature=1.0,
                                                   top_k=None,
                                                   top_p=None,
                                                   eos_token_id=tokenizer.eos_token_id,
                                                   no_repeat_ngram_size=3,
                                                   num_return_sequences=1)

        response = tokenizer.batch_decode(generated_answers_encoded,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)

        output.append(response)

    return SimpleText(dict(text=output))
