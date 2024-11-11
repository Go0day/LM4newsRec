import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention
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

class NewsEncoder(torch.nn.Module):
    def __init__(self, config):
        super(NewsEncoder, self).__init__()
        self.config = config
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
        self.multihead_self_attention = MultiHeadSelfAttention(
            self.dim, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, self.dim)

    def forward(self, news):
        """
        Args:
            news: dictionary with tokenized title
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # Assuming news['title'] contains 'input_ids' and 'attention_mask'
        # print(news)
        # print(news["title"])
        # news_input = {"input_ids": news["title"].to(device), "attention_mask": news["title_mask"].to(device)}
        news_input = {"input_ids": news["title"][:,0].to(device), 
                      "attention_mask": news["title"][:,1].to(device)}

        # news_input = {"input_ids": news["title"].to(device)}
        news_vector = self.bert(**news_input)[0]  # Take all token embeddings
        news_vector = news_vector[:, 0]  # [CLS] token representation
        news_vector = self.pooler(news_vector)
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(multihead_news_vector, p=self.config.dropout_probability, training=self.training)
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector
    
    # def init_weights(self, module):
    #     """Initialize weights for custom layers."""
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 0)
