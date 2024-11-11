# News Recommendation

Welcome to the **Revisiting Language Models in Neural News Recommender Systems** repository. This project provides implementations of various news recommendation models, leveraging language models to deliver news recommendations.

## Models Available

### News Recommendation Models

| **Model** | **Description** | **Reference** |
| --------- | --------------- | ------------- |
| NAML      | Neural News Recommendation with Attentive Multi-View Learning | [Read Paper](https://arxiv.org/abs/1907.05576) |
| NRMS      | Neural News Recommendation with Multi-Head Self-Attention | [Read Paper](https://www.aclweb.org/anthology/D19-1671/) |
| LSTUR     | Neural News Recommendation with Long- and Short-term User Representations | [Read Paper](https://www.aclweb.org/anthology/P19-1033.pdf) |

### Language Models

| **Model** | **Details** |
| --------- | ----------- |
| GloVe     | [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove) |
| BERT      | [Google BERT on Hugging Face](https://huggingface.co/google-bert) |
| BERT (Variants) | [Prajjwal1 BERT Variants](https://huggingface.co/prajjwal1) |
| RoBERTa   | [Facebook AI RoBERTa](https://huggingface.co/FacebookAI) |
| LLaMA     | [Meta LLaMA on Hugging Face](https://huggingface.co/meta-llama) |

## Getting Started

### Step 1: Download and Preprocess the Data

Prepare the data and embeddings before running the models:

```bash
# Create a data directory
mkdir data && cd data

# Download GloVe pre-trained word embeddings (if required)
# wget https://nlp.stanford.edu/data/glove.840B.300d.zip
# sudo apt install unzip
# unzip glove.840B.300d.zip -d glove
# rm glove.840B.300d.zip

# Download the MIND dataset (Microsoft News Dataset)
# By downloading, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977)
# and you can learn more about the dataset at https://msnews.github.io/.

# Uncomment the following lines to use the MIND Large dataset (note: MIND Large test set doesn't have labels)
# wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip \
# https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip \
# https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
# unzip MINDlarge_train.zip -d train
# unzip MINDlarge_dev.zip -d val
# unzip MINDlarge_test.zip -d test
# rm MINDlarge_*.zip

# Uncomment the following lines to use the MIND Small dataset (note: MIND Small doesn't have a test set)
cd data/original/
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip \
https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d val
cp -r val test # MIND Small has no test set
rm MINDsmall_*.zip

# Preprocess data into the appropriate format using the selected language model
# Example: using BERT
cd ..
python3 bert/data_preprocess.py
# Update `num_*` in `bert/config.py` based on the output of `---language_model_path----/data_preprocess.py`
```

### Step 2: Configure the Model

Modify `config.py` to specify the target model. The configuration file includes a general section (applicable to all models) and model-specific settings.

```bash
vim bert/config.py
```

Ensure the correct data paths are set in `config.py` and update the LLM path in `utils.py` as needed.

### Step 3: Train and Evaluate

Run the training and evaluation scripts. Example: using NAML with BERT base version.

```bash
cd bert
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=NAMLbert python train.py --pretrained_mode=bert --word_embedding_dim=768 --learning_rate=0.00001 --dropout_probability=0.2 --batch_size=16 --finetune_layers=4

CUDA_VISIBLE_DEVICES=0 MODEL_NAME=NAMLbert python evaluate.py --pretrained_mode=bert --word_embedding_dim=768 --learning_rate=0.00001 --dropout_probability=0.2 --batch_size=16 --finetune_layers=4
```

## Credits

- **Dataset**: Provided by the Microsoft News Dataset (MIND). Learn more at [MIND](https://msnews.github.io/).
- **News Recommendation Models**: Provided by yusanshi. Repository: [news-recommendation](https://github.com/yusanshi/news-recommendation)
