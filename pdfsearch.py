import os
from flask import Blueprint, Flask, render_template
from flask import request
from werkzeug.utils import secure_filename
import re

pdfsearch=Blueprint("pdfsearch", __name__, static_folder="static", template_folder="templates")

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from collections import OrderedDict

name = "mrm8488/bert-small-finetuned-squadv2"

tokenizer = AutoTokenizer.from_pretrained(name,)

model = AutoModelForQuestionAnswering.from_pretrained(name)

model.to('cuda')

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer` string and tries to identify 
    the words within the `answer` that can answer the question. Prints them out.
    '''
    
    # tokenize the input text and get the corresponding indices
    token_indices = tokenizer.encode(question, answer_text)

    # Search the input_indices for the first instance of the `[SEP]` token.
    sep_index = token_indices.index(tokenizer.sep_token_id)

    seg_one = sep_index + 1

    # The remainders lie in the second segment.
    seg_two = len(token_indices) - seg_one
    
    # Construct the list of 0s and 1s.
    segment_ids = [0]*seg_one + [1]*seg_two

    # get the answer for the question
    start_scores, end_scores = model(torch.tensor([token_indices]), # The tokens representing our input combining question and answer.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer

    # Find the tokens with the highest `start` and `end` scores.
    answer_begin = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    indices_tokens = tokenizer.convert_ids_to_tokens(token_indices)
    
    answer = indices_tokens[answer_begin:answer_end+1]
    #remove special tokens
    answer = [word.replace("▁","") if word.startswith("▁") else word for word in answer] #use this when using model "twmkn9/albert-base-v2-squad2"
    answer = " ".join(answer).replace("[CLS]","").replace("[SEP]","").replace(" ##","")
    
    return answer

class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='mrm8488/bert-small-finetuned-squadv2'):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.model = self.model.to('cuda')
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        self.inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt").to('cuda')
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        """ 
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model. 

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1 # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k,v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)
            
            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks)-1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102]).to('cuda')))
                    else:
                        thing = torch.cat((thing, torch.tensor([1]).to('cuda')))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                a = self.model(**chunk)


                answer_start_scores = a[0]
                answer_end_scores = a[1]
                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + "  "
            return answer
        else:
            a = self.model(**self.inputs)


            answer_start_scores = a[0]
            answer_end_scores = a[1]      
            answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score
        
            return self.convert_ids_to_string(self.inputs['input_ids'][0][
                                              answer_start:answer_end])

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))

reader = DocumentReader("deepset/bert-base-cased-squad2") 

@pdfsearch.route('/uploader', methods = ['GET', 'POST']) 
@pdfsearch.route('/pdfsearch', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      f = request.files['file']
      f.filename = "sample.pdf"
      f.save(f.filename)
      form = request.form
      result = []
    #   bert_abstract = form['paragraph']
      question = form['question']
      text = ""
      for page_layout in extract_pages("sample.pdf"):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                res = element.get_text()
                res1 = re.sub(r'[^\w\s]', ' ', str(res))
                res1 = re.sub(r"^\s+|\s+$", "", res1) # leading and trailing spaces
                res1 = re.sub(' +', ' ', res1) #removing multiple spaces
                text+=res1
      reader.tokenize(question,text)
      result.append(question)
      result.append(reader.get_answer())
      return render_template("pdfsearch.html",result = result)

    return render_template("pdfsearch.html")
