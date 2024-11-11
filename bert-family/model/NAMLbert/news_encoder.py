import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import numpy  as np
import pandas as pd
import csv
from transformers import AutoModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_weights(m: nn.Module):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)  # Corrected init usage
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

class TextEncoder(torch.nn.Module):
    def __init__(self, word_embedding_dim, num_filters,
                 window_size, query_vector_dim, dropout_probability, config):
        super(TextEncoder, self).__init__()
        self.config = config
        self.dropout_probability = dropout_probability
        #  project layer

        bert = AutoModel.from_pretrained(config.pretrained_model_name)
        self.dim = bert.config.hidden_size
        self.bert = bert
        # Freeze all layers except the last `config.finetune_layers` layers
        num_layers = len(self.bert.encoder.layer)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < num_layers - config.finetune_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        # for param in self.bert.parameters():
        #     print(param.requires_grad)
        self.pooler = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(0.1),  # Consider tuning this dropout rate
            nn.LayerNorm(self.dim),
            nn.SiLU(),
        )
        self.pooler.apply(init_weights)


        self.CNN = nn.Conv2d(1,
                             num_filters, (window_size, word_embedding_dim),
                             padding=(int((window_size - 1) / 2), 0))
        self.additive_attention = AdditiveAttention(query_vector_dim,
                                                    num_filters)

    def forward(self, text):
        # batch_size, num_words_text, word_embedding_dim
        news_input = {"input_ids": text[:,0].to(device), 
                      "attention_mask": text[:,1].to(device)}
        # news_input = {"input_ids": news["title"].to(device)}
        text_vector = self.bert(**news_input)[0]  # Take all token embeddings
        text_vector = text_vector[:, 0]  # [CLS] token representation
        text_vector = self.pooler(text_vector)
        # batch_size, num_filters, num_words_title
        # convoluted_text_vector = self.CNN(
        #     text_vector.unsqueeze(dim=1).float()).squeeze(dim=3)
        convoluted_text_vector = self.CNN(
            text_vector.unsqueeze(1).unsqueeze(2).float()).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_text_vector = F.dropout(F.relu(convoluted_text_vector),
                                          p=self.dropout_probability,
                                          training=self.training)

        # batch_size, num_filters
        text_vector = self.additive_attention(
            activated_text_vector.transpose(1, 2))
        return text_vector


class ElementEncoder(torch.nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class NewsEncoder(torch.nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
        assert len(config.dataset_attributes['news']) > 0
        text_encoders_candidates = ['title', 'abstract']
        self.text_encoders = nn.ModuleDict({
            name:
            TextEncoder(config.word_embedding_dim,
                        config.num_filters, config.window_size,
                        config.query_vector_dim, config.dropout_probability, config)
            for name in (set(config.dataset_attributes['news'])
                         & set(text_encoders_candidates))
        })
        category_embedding = nn.Embedding(config.num_categories,
                                          config.category_embedding_dim,
                                          padding_idx=0)
        element_encoders_candidates = ['category', 'subcategory']
        self.element_encoders = nn.ModuleDict({
            name:
            ElementEncoder(category_embedding, config.category_embedding_dim,
                           config.num_filters)
            for name in (set(config.dataset_attributes['news'])
                         & set(element_encoders_candidates))
        })
        if len(config.dataset_attributes['news']) > 1:
            self.final_attention = AdditiveAttention(config.query_vector_dim,
                                                     config.num_filters)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract,
                }
        Returns:
            (shape) batch_size, num_filters
        """
        text_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.text_encoders.items()
        ]
        element_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))
        return final_news_vector
