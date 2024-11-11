import datetime
import importlib
import os
import random
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from texttable import Texttable
import json
import pickle
from nltk.tokenize import word_tokenize

import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from functools import wraps

from config import model_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NAML_like = ['NAML', 'NAMLlinear', 'NAMLtext', 'NAMLbert']

 

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


def timer(func):
    """The timer decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("%s function cost: %fs" % (func.__name__, end_time - start_time))
        return result
    return wrapper


@timer
def pretrained_encode_bert(news_dataframe, repeat_times_title, repeat_times_abstract):
    bert_path = "DOWNLOAD_PATH/llms/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_path, legacy=False, clean_up_tokenization_spaces=True)
    model = BertModel.from_pretrained(bert_path)
    model = model.to(device)
    model.eval()  # Freeze BERT model

    def encode_text(texts, repeat_times):
        # Prepare a list to store the embeddings
        all_embeddings = []
        # Process texts in batches of 100
        for i in range(0, len(texts), 100):
            batch_texts = texts[i:i + 100]
            encoded_inputs = tokenizer(batch_texts.tolist(), padding=True, truncation=True, max_length=repeat_times, return_tensors='pt')
            encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            embeddings = outputs.last_hidden_state  # Extract the embeddings for the [CLS] token
            assert outputs.last_hidden_state.shape[1] == repeat_times
            # repeated_embeddings = np.repeat(embeddings.cpu().numpy(), repeat_times, axis=0).reshape(-1, repeat_times, embeddings.size(1))
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings

    # Encode 'title' and 'abstract' in batches
    if model_name in NAML_like:
        news_dataframe['title_encoded'] = encode_text(news_dataframe['title'], repeat_times_title)
        news_dataframe['abstract_encoded'] = encode_text(news_dataframe['abstract'], repeat_times_abstract)
        # Create a new DataFrame to hold encoded data
        news_dataframe_encoded = news_dataframe.drop(['title', 'abstract'], axis=1)
        news_dataframe_encoded['title'] = pd.Series(news_dataframe['title_encoded'], index=news_dataframe.index)
        news_dataframe_encoded['abstract'] = pd.Series(news_dataframe['abstract_encoded'], index=news_dataframe.index)
        # Optionally, drop the old columns if they were added before renaming
        news_dataframe_encoded.drop(['title_encoded', 'abstract_encoded'], axis=1, inplace=True)
    else:
        news_dataframe['title_encoded'] = encode_text(news_dataframe['title'], repeat_times_title)
        # Create a new DataFrame to hold encoded data
        news_dataframe_encoded = news_dataframe.drop(['title'], axis=1)
        news_dataframe_encoded['title'] = pd.Series(news_dataframe['title_encoded'], index=news_dataframe.index)
        # Optionally, drop the old columns if they were added before renaming
        news_dataframe_encoded.drop(['title_encoded'], axis=1, inplace=True)
    del model
    torch.cuda.empty_cache()
    
    return news_dataframe_encoded


def load_glove_model(glove_path):
    print("Loading Glove Model")
    # Load the embeddings using pandas read_table with space separator and no header
    glove_data = pd.read_table(
        glove_path,
        sep=" ",
        header=None,
        quoting=csv.QUOTE_NONE,
        index_col=0
    )
    
    # Convert the DataFrame to a dictionary with word as keys and numpy arrays as values
    glove_model = {index: row.values for index, row in glove_data.iterrows()}
    print("Done.", len(glove_model), " words loaded!")
    
    return glove_model

@timer
def pretrained_encode_glove(news_dataframe, num_title, num_abstract):
    # Load GloVe embeddings
    glove_path = "DOWNLOAD_PATH/llms/glove/glove.840B.300d.txt"
    glove_model = load_glove_model(glove_path)

    def encode_text(texts, num_words2Encode):
        encoded_texts = []
        for text in texts:
            # Tokenize the text and take up to the first 20 words
            tokens = word_tokenize(text)[:num_words2Encode]
            embeddings = []
            for token in tokens:
                if token.lower() in glove_model:
                    embeddings.append(glove_model[token.lower()])
                else:
                    embeddings.append(np.zeros(300))  # Append zero vector if token not found
            # If fewer than 20 words, pad with zero-vectors
            while len(embeddings) < num_words2Encode:
                embeddings.append(np.zeros(300))
            encoded_texts.append(np.array(embeddings))
        return np.array(encoded_texts)

    # Encode 'title' and 'abstract'
    if model_name in NAML_like:
        news_dataframe['title'] = list(encode_text(news_dataframe['title'], num_title))
        news_dataframe['abstract'] = list(encode_text(news_dataframe['abstract'], num_abstract))
    else:
        news_dataframe['title'] = list(encode_text(news_dataframe['title'], num_title))
    return news_dataframe

# @timer
# def pretrained_encode_llama(news_dataframe, repeat_times_title, repeat_times_abstract):

#     # Load LLaMA model and tokenizer
#     model_id = "DOWNLOAD_PATH/llms/Meta-Llama-3.1-8B-Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     tokenizer.pad_token = tokenizer.eos_token
#     model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
#     model = model.to(device)
#     model.eval()

#     def encode_text(texts, repeat_times):
#         all_embeddings = []
#         batch_size = 10  # Adjusted smaller batch size if necessary
#         for i in range(0, len(texts), batch_size):
#             batch_texts = texts[i:i + batch_size]
#             prompt_template = "This news: {text} means in one word:"
#             formatted_texts = [prompt_template.format(text=x) for x in batch_texts]

#             t_input = tokenizer(formatted_texts, padding=True, return_tensors="pt")
#             t_input = {key: value.to(device) for key, value in t_input.items()}
#             with torch.no_grad():
#                 outputs = model(**t_input, output_hidden_states=True, return_dict=True)
#                 last_hidden_state = outputs.hidden_states[-1]

#             idx_of_the_last_non_padding_token = t_input['attention_mask'].bool().sum(1) - 1
#             sentence_embeddings = last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx_of_the_last_non_padding_token]
#             sentence_embeddings = sentence_embeddings.detach()  # Detach embeddings

#             repeated_embeddings = np.repeat(sentence_embeddings.cpu().numpy(), repeat_times, axis=0).reshape(-1, repeat_times, sentence_embeddings.shape[-1])
#             all_embeddings.extend(repeated_embeddings)

#             del outputs, last_hidden_state, sentence_embeddings, t_input  # Free memory
#             torch.cuda.empty_cache()  # Clear cache after each batch

#         return all_embeddings
#     if model_name in NAML_like:
#         news_dataframe['title_encoded'] = encode_text(news_dataframe['title'], repeat_times_title)
#         news_dataframe['abstract_encoded'] = encode_text(news_dataframe['abstract'], repeat_times_abstract)

#         news_dataframe_encoded = news_dataframe.drop(['title', 'abstract'], axis=1)
#         news_dataframe_encoded['title'] = pd.Series(news_dataframe['title_encoded'], index=news_dataframe.index)
#         news_dataframe_encoded['abstract'] = pd.Series(news_dataframe['abstract_encoded'], index=news_dataframe.index)
#         news_dataframe_encoded.drop(['title_encoded', 'abstract_encoded'], axis=1, inplace=True)
#     else:
#         news_dataframe['title_encoded'] = encode_text(news_dataframe['title'], repeat_times_title)

#         news_dataframe_encoded = news_dataframe.drop(['title'], axis=1)
#         news_dataframe_encoded['title'] = pd.Series(news_dataframe['title_encoded'], index=news_dataframe.index)
#         news_dataframe_encoded.drop(['title_encoded'], axis=1, inplace=True)

#     # del model
#     # torch.cuda.empty_cache()

#     return news_dataframe_encoded 


@timer
def pretrained_encode_llama(news_dataframe, repeat_times_title, repeat_times_abstract):

    # Load LLaMA model and tokenizer
    model_path = "DOWNLOAD_PATH/llms/Meta-Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                device_map='auto',
                                                output_hidden_states=True,
                                                trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0  # Set the padding token.
    tokenizer.padding_side = "left"  # Allow batched inference

    def encode_text(texts, repeat_times):
        all_embeddings = []
        batch_size = 10  # Adjusted smaller batch size if necessary
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            if config.with_prompt==1:
                prompt_template = "This news: {text} means in one word:"
                formatted_texts = [prompt_template.format(text=x) for x in batch_texts]
            else:
                formatted_texts = [x for x in batch_texts]

            t_input = tokenizer.batch_encode_plus(
                formatted_texts,
                return_tensors='pt',
                padding=True,
                max_length=repeat_times,
                truncation=repeat_times is not None
            )
            
            t_input = {key: value.to(device) for key, value in t_input.items()}
            with torch.no_grad():
                hidden_states = model(output_hidden_states=True, return_dict=True, **t_input).hidden_states
                last_layer = hidden_states[-1]
                if last_layer.shape[1] < repeat_times:

                    pad_size = repeat_times - last_layer.shape[1]
                    # Create a padding tensor of zeros
                    padding_tensor = torch.zeros(last_layer.shape[0], pad_size, last_layer.shape[2]).to(device)
                    # Concatenate the original tensor with the padding tensor along the channel dimension
                    outputs = torch.cat((last_layer, padding_tensor), dim=1)
                else:
                    outputs = last_layer
                assert outputs.shape[1] == repeat_times
                if outputs.dtype == torch.bfloat16:
                    # bfloat16 not support for .numpy()
                    outputs = outputs.float()

            all_embeddings.extend(outputs.cpu().numpy())

        return all_embeddings
    if model_name in NAML_like:
        news_dataframe['title_encoded'] = encode_text(news_dataframe['title'], repeat_times_title)
        news_dataframe['abstract_encoded'] = encode_text(news_dataframe['abstract'], repeat_times_abstract)

        news_dataframe_encoded = news_dataframe.drop(['title', 'abstract'], axis=1)
        news_dataframe_encoded['title'] = pd.Series(news_dataframe['title_encoded'], index=news_dataframe.index)
        news_dataframe_encoded['abstract'] = pd.Series(news_dataframe['abstract_encoded'], index=news_dataframe.index)
        news_dataframe_encoded.drop(['title_encoded', 'abstract_encoded'], axis=1, inplace=True)
    else:
        news_dataframe['title_encoded'] = encode_text(news_dataframe['title'], repeat_times_title)

        news_dataframe_encoded = news_dataframe.drop(['title'], axis=1)
        news_dataframe_encoded['title'] = pd.Series(news_dataframe['title_encoded'], index=news_dataframe.index)
        news_dataframe_encoded.drop(['title_encoded'], axis=1, inplace=True)

    del model
    del tokenizer
    
    torch.cuda.empty_cache()

    return news_dataframe_encoded 


def init_seed(seed, reproducibility=True):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False



def save_gpt_news(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_gpt_news(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def save_news_freq(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_news_freq(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_news_dataset(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_news_dataset(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data



