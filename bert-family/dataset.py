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
import ast

from utils import pretrained_encode_bert, pretrained_encode_glove, pretrained_encode_llama

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class BaseDataset(Dataset):
#     def __init__(self, behaviors_path, news_path, config):
#         super(BaseDataset, self).__init__()
#         self.config = config
#         assert all(attribute in [
#             'category', 'subcategory', 'title', 'abstract', 'title_entities',
#             'abstract_entities'
#         ] for attribute in config.dataset_attributes['news'])
#         assert all(attribute in ['user', 'clicked_news_length']
#                    for attribute in config.dataset_attributes['record'])

#         self.behaviors_parsed = pd.read_table(behaviors_path)

#         self.news_parsed = pd.read_table(
#             news_path,
#             index_col='id',
#             usecols=['id'] + config.dataset_attributes['news'],
#             converters={
#                 attribute: literal_eval
#                 for attribute in set(config.dataset_attributes['news']) & set([
#                     'title_entities', 'abstract_entities'
#                     ])
#             })
#         self.news_parsed['title_mask'] = 0
#         self.news_parsed['abstract_mask'] = 0


#         self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
#         self.news2dict = self.news_parsed.to_dict('index')


#         for key1 in self.news2dict.keys():
#             keys_to_iterate = list(self.news2dict[key1].keys())
#             if 'title_mask' not in keys_to_iterate:
#                 print("========================================================================")
#             for key2 in keys_to_iterate:
#                 if key2 in ['title', 'abstract']:
#                     self.news2dict[key1][key2] = ast.literal_eval(self.news2dict[key1][key2])
#                     assert torch.tensor(self.news2dict[key1][key2]['input_ids']).shape == torch.tensor(self.news2dict[key1][key2]['attention_mask']).shape
#                     # # self.news2dict[key1][key2] = torch.cat([torch.tensor(self.news2dict[key1][key2]['input_ids']).unsqueeze(0), torch.tensor(self.news2dict[key1][key2]['attention_mask']).unsqueeze(0)], dim=0)
#                     # self.news2dict[key1][key2] = torch.stack([torch.tensor(self.news2dict[key1][key2]['input_ids']), torch.tensor(self.news2dict[key1][key2]['attention_mask'])])

#                     if key2 == 'title':
#                         self.news2dict[key1]['title_mask'] = torch.tensor(self.news2dict[key1][key2]['attention_mask'])
#                     elif key2 == 'abstract':
#                         self.news2dict[key1]['abstract_mask'] = torch.tensor(self.news2dict[key1][key2]['attention_mask'])
#                     self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2]['input_ids'])

#                 elif key2 in ['title_mask', 'abstract_mask']:
#                     pass
#                 else:
#                     self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2])

#         padding_all = {
#             'category': 0,
#             'subcategory': 0,
#             'title': [0] * config.num_words_title,  # Keeping title as an empty string for consistency in handling text
#             'abstract': [0] * config.num_words_abstract,  # Same for abstract
#             'title_entities': [0] * config.num_words_title,  # Assuming these are numerical lists
#             'abstract_entities': [0] * config.num_words_abstract,
#             'title_mask': [0] * config.num_words_title,
#             'abstract_mask': [0] * config.num_words_abstract,
#         }
#         for key in padding_all.keys():
#             padding_all[key] = torch.tensor(padding_all[key])

#         self.padding = {
#             k: v
#             for k, v in padding_all.items()
#             if k in config.dataset_attributes['news']
#         }
    
#     def __len__(self):
#         return len(self.behaviors_parsed)

#     def __getitem__(self, idx):
#         item = {}
#         row = self.behaviors_parsed.iloc[idx]
#         if 'user' in config.dataset_attributes['record']:
#             item['user'] = row.user
#         item["clicked"] = list(map(int, row.clicked.split()))
#         item["candidate_news"] = [
#             self.news2dict[x] for x in row.candidate_news.split()
#         ]
#         item["clicked_news"] = [
#             self.news2dict[x]
#             for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
#         ]
#         if 'clicked_news_length' in config.dataset_attributes['record']:
#             item['clicked_news_length'] = len(item["clicked_news"])
#         repeated_times = config.num_clicked_news_a_user - \
#             len(item["clicked_news"])
#         assert repeated_times >= 0
#         item["clicked_news"] = [self.padding
#                                 ] * repeated_times + item["clicked_news"]

#         return item



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

        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')


        for key1 in self.news2dict.keys():
            keys_to_iterate = list(self.news2dict[key1].keys())
            for key2 in keys_to_iterate:
                if key2 in ['title', 'abstract']:
                    self.news2dict[key1][key2] = ast.literal_eval(self.news2dict[key1][key2])
                    assert torch.tensor(self.news2dict[key1][key2]['input_ids']).shape == torch.tensor(self.news2dict[key1][key2]['attention_mask']).shape
                    self.news2dict[key1][key2] = torch.cat([torch.tensor(self.news2dict[key1][key2]['input_ids']).unsqueeze(0), torch.tensor(self.news2dict[key1][key2]['attention_mask']).unsqueeze(0)], dim=0)

                else:
                    self.news2dict[key1][key2] = torch.tensor(self.news2dict[key1][key2])

        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [tokens + [0] * (config.num_words_title - 2) for tokens in [[101, 102], [1, 1]]],  # Keeping title as an empty string for consistency in handling text
            'abstract': [tokens + [0] * (config.num_words_abstract - 2) for tokens in [[101, 102], [1, 1]]],  # Same for abstract
            'title_entities': [0] * config.num_words_title,  # Assuming these are numerical lists
            'abstract_entities': [0] * config.num_words_abstract,
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
        clicked_times = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]
        item["clicked_news_mask"] = [0] * repeated_times + [1] * clicked_times
        return item