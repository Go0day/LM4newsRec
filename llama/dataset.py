from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from os import path
import numpy as np
from config import model_name
import importlib
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pretrained_encode_bert, pretrained_encode_glove, pretrained_encode_llama

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NAML_like = ['NAML', 'NAMLlinear', 'NAMLtext']


class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, config):
        super(BaseDataset, self).__init__()
        self.config = config
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities'
        ] for attribute in config.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes['record'])

        self.behaviors_parsed = pd.read_table(behaviors_path)

        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title_entities', 'abstract_entities'
                    ])
            })

        # encode the news embedding
        if config.pretrained_mode == 'bert':
            self.news_parsed_encoded = pretrained_encode_bert(self.news_parsed, repeat_times_title=config.num_words_title, repeat_times_abstract=config.num_words_abstract)
        elif config.pretrained_mode == 'glove':
            self.news_parsed_encoded = pretrained_encode_glove(self.news_parsed, num_title=config.num_words_title, num_abstract=config.num_words_abstract)
        elif config.pretrained_mode == 'llama':
            self.news_parsed_encoded = pretrained_encode_llama(self.news_parsed, repeat_times_title=config.num_words_title, repeat_times_abstract=config.num_words_abstract, llama_mode=config.llama_mode)
        
        del self.news_parsed

        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed_encoded.index)}
        self.news2dict = self.news_parsed_encoded.to_dict('index')

        del self.news_parsed_encoded

        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2])
        num_dim_title = config.num_words_title
        if model_name in NAML_like:
            num_dim_title = 5
        else:
            num_dim_title = 10
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [[0.0]* config.word_embedding_dim] * num_dim_title,  # Keeping title as an empty string for consistency in handling text
            'abstract': [[0.0]* config.word_embedding_dim] * num_dim_title,  # Same for abstract
            'title_entities': [0] * config.num_words_title,  # Assuming these are numerical lists
            'abstract_entities': [0] * config.num_words_abstract
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

        
    
    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            self.news2dict[x] for x in row.candidate_news.split()
        ]
        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]

        return item
