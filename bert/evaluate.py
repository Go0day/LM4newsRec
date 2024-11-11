import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
from multiprocessing import Pool
import os
from utils import pretrained_encode_bert, pretrained_encode_glove, pretrained_encode_llama
from utils import save_news_dataset, load_news_dataset
import ast

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def should_display_progress():
    return sys.stdout.isatty()

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}

class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title_entities', 'abstract_entities'
                ])
            })

        self.news2dict = self.news_parsed.to_dict('index')

        for key1 in self.news2dict.keys():
            keys_to_iterate = list(self.news2dict[key1].keys())
            for key2 in keys_to_iterate:
                if key2 in ['title', 'abstract']:
                    self.news2dict[key1][key2] = ast.literal_eval(self.news2dict[key1][key2])
                    assert torch.tensor(self.news2dict[key1][key2]['input_ids']).shape == torch.tensor(self.news2dict[key1][key2]['attention_mask']).shape
                    self.news2dict[key1][key2] = torch.cat([torch.tensor(self.news2dict[key1][key2]['input_ids']).unsqueeze(0), torch.tensor(self.news2dict[key1][key2]['attention_mask']).unsqueeze(0)], dim=0)
                elif type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news2dict)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item

class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        # self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        if model_name == 'LSTUR' or model_name == 'LSTURlinear' or model_name== 'LSTURbert':
            print(f'User miss rate: {user_missed/user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        # self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


@torch.no_grad()
def evaluate_popular(model, directory, num_workers, news_dataset_built=None, max_count=sys.maxsize, num_groups=5):
    """
    Evaluate model on target directory, output user group AUC, MRR, nDCG@5, nDCG@10 based on interaction popularity.
    
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        num_workers: processes number for calculating metrics
        num_groups: Number of groups to divide users by popularity (default 5 means 0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
    Returns:
        A dictionary containing evaluation metrics for each user group
    """
    if news_dataset_built:
        news_dataset = news_dataset_built
    else:
        news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    news2vector = {}
    progress = news_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating vectors for news")
    for minibatch in progress:
        news_ids = minibatch["id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(list(news2vector.values())[0].size())

    user_dataset = UserDataset(path.join(directory, 'behaviors.tsv'),
                               path.join(config.original_data_path, 'train/user2int.tsv'))
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    # Calculate user interaction counts (clicked_news_length directly gives the number of clicked news)
    user_interaction_count = {}
    for minibatch in user_dataloader:
        for user_string, clicked_length in zip(minibatch["clicked_news_string"], minibatch["clicked_news_length"]):
            user_interaction_count[user_string] = clicked_length.item()

    # Define percentile groups
    sorted_users = sorted(user_interaction_count.items(), key=lambda x: x[1])
    percentiles = [20, 40, 60, 80, 100]
    user_percentiles = np.percentile([x[1] for x in sorted_users], percentiles)

    # Create user groups
    user_groups = {
        '0-20%': [],
        '20-40%': [],
        '40-60%': [],
        '60-80%': [],
        '80-100%': [],
        '0-50%': [],
        '50-100%': [],
        'Cold (≤ 5 clicks)': [],
        'Heavy (≥ 5 clicks)': []
    }

    for user, count in user_interaction_count.items():
        if count <= 5:
            user_groups["Cold (≤ 5 clicks)"].append(user)
        else:
            user_groups["Heavy (≥ 5 clicks)"].append(user)

        if count <= user_percentiles[0]:
            user_groups['0-20%'].append(user)
        elif count <= user_percentiles[1]:
            user_groups['20-40%'].append(user)
        elif count <= user_percentiles[2]:
            user_groups['40-60%'].append(user)
        elif count <= user_percentiles[3]:
            user_groups['60-80%'].append(user)
        else:
            user_groups['80-100%'].append(user)

        # Create 0-50% and 50-100% groups
        if count <= np.percentile([x[1] for x in sorted_users], 50):
            user_groups['0-50%'].append(user)
        else:
            user_groups['50-100%'].append(user)

    # Initialize metrics dictionary
    group_metrics = {group: {"auc": [], "mrr": [], "ndcg5": [], "ndcg10": []} for group in user_groups}

    # Initialize user2vector dictionary
    user2vector = {}
    progress = user_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating vectors for users")
    
    for minibatch in progress:
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list], dim=0) 
                for news_list in minibatch["clicked_news"]
            ], dim=0).transpose(0, 1)
            
            if model_name in ['LSTUR', 'LSTURlinear', 'LSTURbert', 'LSTURllama']:
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'], clicked_news_vector
                )
            else:
                user_vector = model.get_user_vector(clicked_news_vector)
                
            for user_string, vector in zip(user_strings, user_vector):
                if user_string not in user2vector:
                    user2vector[user_string] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0
    progress = behaviors_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating probabilities")

    # Calculate metrics for each user group
    for minibatch in progress:
        count += 1
        if count == max_count:
            break

        candidate_news_vector = torch.stack([news2vector[news[0].split('-')[0]] for news in minibatch['impressions']], dim=0)
        user_string = minibatch['clicked_news_string'][0]
        if user_string not in user2vector:
            continue

        user_vector = user2vector[user_string]
        click_probability = model.get_prediction(candidate_news_vector, user_vector)
        y_pred = click_probability.tolist()
        y_true = [int(news[0].split('-')[1]) for news in minibatch['impressions']]

        for group, users in user_groups.items():
            if user_string in users:
                auc, mrr, ndcg5, ndcg10 = calculate_single_user_metric((y_true, y_pred))
                group_metrics[group]["auc"].append(auc)
                group_metrics[group]["mrr"].append(mrr)
                group_metrics[group]["ndcg5"].append(ndcg5)
                group_metrics[group]["ndcg10"].append(ndcg10)
    
    # Calculate average metrics for each group
    final_metrics = {}
    for group, metrics in group_metrics.items():
        final_metrics[group] = {
            "auc": np.nanmean(metrics["auc"]),
            "mrr": np.nanmean(metrics["mrr"]),
            "ndcg5": np.nanmean(metrics["ndcg5"]),
            "ndcg10": np.nanmean(metrics["ndcg10"]),
            "user_count": len(user_groups[group])
        }

    # Print the results
    for group, metrics in final_metrics.items():
        print(f"Group {group}: User Count = {metrics['user_count']}")
        print(f"AUC: {metrics['auc']:.4f}, MRR: {metrics['mrr']:.4f}, nDCG@5: {metrics['ndcg5']:.4f}, nDCG@10: {metrics['ndcg10']:.4f}\n")

    return final_metrics

@torch.no_grad()
def evaluate(model, directory, num_workers, news_dataset_built=None, max_count=sys.maxsize):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        num_workers: processes number for calculating metrics
    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    if news_dataset_built:
        news_dataset = news_dataset_built
    else:
        news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    news2vector = {}
    progress = news_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating vectors for news")
    for minibatch in progress:
        news_ids = minibatch["id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())

    user_dataset = UserDataset(path.join(directory, 'behaviors.tsv'),
                               path.join(config.original_data_path, 'train/user2int.tsv'))
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    progress = user_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating vectors for users")
    for minibatch in progress:
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                              dim=0).transpose(0, 1)
            if model_name == 'LSTUR' or model_name == 'LSTURlinear' or model_name== 'LSTURbert':
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'],
                    clicked_news_vector)
            else:
                user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0

    tasks = []
    progress = behaviors_dataloader
    if should_display_progress():
        progress = tqdm(progress, desc="Calculating probabilities")
    for minibatch in progress:
        count += 1
        if count == max_count:
            break

        candidate_news_vector = torch.stack([
            news2vector[news[0].split('-')[0]]
            for news in minibatch['impressions']
        ],
                                            dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = model.get_prediction(candidate_news_vector,
                                                 user_vector)

        y_pred = click_probability.tolist()
        y_true = [
            int(news[0].split('-')[1]) for news in minibatch['impressions']
        ]

        tasks.append((y_true, y_pred))

    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(
        ndcg10s)


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(os.path.join(config.current_data_path + '/checkpoint', config.pretrained_mode, model_name))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if config.pretrained_mode == 'llama':
        dataset_evaluate_path = os.path.join(config.current_data_path + '/tmp_pkl',config.pretrained_mode, model_name + "/evaluate_dataset" + str(config.with_prompt) + ".pkl")
    else:
        dataset_evaluate_path = os.path.join(config.current_data_path + '/tmp_pkl',config.pretrained_mode, model_name + "/evaluate_dataset.pkl")
        
    if os.path.exists(dataset_evaluate_path):
        evaluate_dataset = load_news_dataset(dataset_evaluate_path)
        auc, mrr, ndcg5, ndcg10 = evaluate(model, path.join(config.original_data_path, 'test'),
                                    config.num_workers, news_dataset_built=evaluate_dataset)
    else:
        auc, mrr, ndcg5, ndcg10 = evaluate(model, path.join(config.original_data_path, 'test'),
                                        config.num_workers)
    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    )
