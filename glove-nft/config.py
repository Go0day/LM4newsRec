import os
import argparse

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'NAML'

# Currently included model
assert model_name in [
    'NRMS', 'NAML', 'LSTUR', 'DKN', 'HiFiArk', 'TANR', 'Exp1', 'NAMLlinear', 'NRMSlinear', 'LSTURlinear'
]



# Set up argument parser
parser = argparse.ArgumentParser(description="Set configuration for the model")
parser.add_argument('--num_categories', type=int, default=1 + 274,
                    help='`num_categories` in `src/config.py` into 1 + {len(category2int)}')
parser.add_argument('--num_words', type=int, default=1 + 70972,
                    help='`num_words` in `src/config.py` into 1 + {len(word2int)}')
parser.add_argument('--num_entities', type=int, default=1 + 12957,
                    help='`num_entities` in `src/config.py` into 1 + {len(entity2int)}')
parser.add_argument('--measure_way', type=str, default='item_rank',
                    help='`measure_way`, could be softmax, rank, and inverse_rank')
parser.add_argument('--exclude_interacted_item', type=int, default=0,
                    help='whether analysis item in user history')
parser.add_argument('--using_UNK', type=int, default=0,
                    help='whether retrain the whole model.')
parser.add_argument('--api_mode', type=str, default='llama3_attack_dpo_lora_1',
                    help='the specific name of api mode -> e.g., gpt-3.5-turbo, gpt-4 or llama3_attack_dpo_lora_1')
parser.add_argument('--attack_mode', type=str, default='llm',
                    help='llm or Copycat, etc. ')
parser.add_argument('--current_data_path', type=str, default='DOWNLOAD_PATH/FOLDER/news-recommendation/data',
                    help='llm or Copycat, etc. ')
parser.add_argument('--attack_folder', type=str, default='DOWNLOAD_PATH/FOLDER/news-recommendation/data/attack',
                    help='llm or Copycat, etc. ')
parser.add_argument('--gpt4_attack_text_path', type=str, default="/data/zhaoyy/news_data/data/gpt_data/test/data_at_interaction_598.json",
                    help='llm or Copycat, etc. ')
parser.add_argument('--attack_folder_mode', type=str, default="test",
                    help='train or test, specific to attack_train or attack_test')
parser.add_argument('--topN', type=int, default=10,
                    help='Evaluate top N Rec Items')
parser.add_argument('--pretrained_mode', type=str, default='bert',
                    help='text encoder for news recommender, could be (glove, bert, or llama)')
parser.add_argument('--original_data_path', type=str, default='DOWNLOAD_PATH/FOLDER/news-recommendation/data/original',
                    help='model train data path')
parser.add_argument('--word_embedding_dim', type=int, default=768,
                    help='golve - 300, bert - 768, llama - 4096')
parser.add_argument('--maintain_word_embedding_dim', type=int, default=300,
                    help='golve - 300, bert - 768, llama - 4096')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--dropout_probability', type=float, default=0.2,
                    help='dropout_probability')
parser.add_argument('--num_attention_heads', type=int, default=15,
                    help='num_attention_heads')
parser.add_argument('--with_prompt', type=int, default=1,
                    help='For LLaMa, with prompt = 1, or 0')
parser.add_argument('--negative_sampling_ratio', type=int, default=2,
                    help='negative sample ratio')
args = parser.parse_args()


class BaseConfig():
    """
    General configurations appiled to all models
    """
    with_prompt = args.with_prompt
    maintain_word_embedding_dim = args.maintain_word_embedding_dim
    topN = args.topN
    current_data_path = args.current_data_path
    measure_way = args.measure_way
    exclude_interacted_item = args.exclude_interacted_item
    analysis_news_popular = 1
    gpt4_attack_text_path = args.gpt4_attack_text_path
    original_data_path = args.original_data_path
    pretrained_mode = args.pretrained_mode
    home_folder = '/data/zhaoyy/news_data'
    seed = 2024
    poisoning_ratio = 0.01
    attack_folder = args.attack_folder
    attack_folder_mode = args.attack_folder_mode
    api_mode = args.api_mode
    using_UNK = args.using_UNK   # if Ture, then the NRS don't need to retraining; else, the NRS couldn't process OOV, and would be better to retraining the NRS and formulate a new Vocabulary list.
    attack_mode = 'llm'
    
    num_epochs = 100
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 1000
    batch_size = 32
    learning_rate = args.learning_rate
    num_workers = 1  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = args.negative_sampling_ratio  # K
    dropout_probability = args.dropout_probability
    # Modify the following by the output of `src/dataprocess.py`
    num_words = args.num_words
    num_categories = args.num_categories
    num_entities = args.num_entities
    num_users = 1 + 50000
    word_embedding_dim = args.word_embedding_dim
    linear_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = args.num_attention_heads

class NRMSlinearConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = args.num_attention_heads

class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3

class NAMLlinearConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 300
    window_size = 3
    long_short_term_method = 'ini'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5

class LSTURlinearConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 300
    window_size = 3
    long_short_term_method = 'ini'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5
    regularizer_loss_weight = 0.1


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1


class Exp1Config(BaseConfig):
    dataset_attributes = {
        # TODO ['category', 'subcategory', 'title', 'abstract'],
        "news": ['category', 'subcategory', 'title'],
        "record": []
    }
    # For multi-head self-attention
    num_attention_heads = 12
    ensemble_factor = 1  # Not use ensemble since it's too expensive
