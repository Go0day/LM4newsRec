from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from os import path
from pathlib import Path
from evaluate import evaluate, NewsDataset, evaluate_popular
import importlib
import datetime
import json
import sys
import wandb
import random
from utils import save_news_dataset, load_news_dataset

try:
    start_time = time.time()
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    wandb.init(
        project = "news",
        config={
            "learning_rate": config.learning_rate,
            "model": model_name,
            "pretrained_mode": config.pretrained_mode,
            "word_embedding_dim": config.word_embedding_dim,
            "dropout_probability": config.dropout_probability
        }
    )
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return total_params, trainable_params

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better

def should_display_progress():
    return sys.stdout.isatty()

def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
        if len(x) <20
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def train():
    writer = SummaryWriter(
        log_dir=
        f"./runs/{model_name}/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if not os.path.exists(os.path.join(config.current_data_path + '/checkpoint',config.pretrained_mode, model_name)):
        os.makedirs(os.path.join(config.current_data_path + '/checkpoint',config.pretrained_mode, model_name))

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load(config.current_data_path + 'train/pretrained_word_embedding.npy')).float()
    except:
        pretrained_word_embedding = None

    if model_name == 'DKN':
        try:
            pretrained_entity_embedding = torch.from_numpy(
                np.load(
                    config.current_data_path + '/train/pretrained_entity_embedding.npy')).float()
        except:
            pretrained_entity_embedding = None

        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(
                    config.current_data_path + '/train/pretrained_context_embedding.npy')).float()
        except:
            pretrained_context_embedding = None

        model = Model(config, pretrained_word_embedding,
                      pretrained_entity_embedding,
                      pretrained_context_embedding).to(device)
    elif model_name == 'Exp1':
        models = nn.ModuleList([
            Model(config, pretrained_word_embedding).to(device)
            for _ in range(config.ensemble_factor)
        ])
    else:
        model = Model(config).to(device)

    if model_name != 'Exp1':
        print(model)
    else:
        print(models[0])

    # dataset = BaseDataset(config.original_data_path + '/train/behaviors_parsed.tsv',
    #                       config.original_data_path + '/train/news_parsed.tsv')
    print("load original data:")
    print(time_since(start_time))
    # if config.pretrained_mode == 'llama':
    #     dataset_train_path = os.path.join(config.current_data_path + '/tmp_pkl',config.pretrained_mode, model_name + "/train_dataset" + str(config.with_prompt) + ".pkl")
    #     dataset_evaluate_path = os.path.join(config.current_data_path + '/tmp_pkl',config.pretrained_mode, model_name + "/evaluate_dataset" + str(config.with_prompt) + ".pkl")
    # else:
    #     dataset_train_path = os.path.join(config.current_data_path + '/tmp_pkl',config.pretrained_mode, model_name + "/train_dataset.pkl")
    #     dataset_evaluate_path = os.path.join(config.current_data_path + '/tmp_pkl',config.pretrained_mode, model_name + "/evaluate_dataset.pkl")
    
    # if os.path.exists(dataset_train_path):
    #     dataset = load_news_dataset(dataset_train_path)
    # else:
    #     dataset = BaseDataset(config.original_data_path + '/train/behaviors_parsed.tsv',
    #                         config.original_data_path + '/train/news_parsed.tsv', 
    #                         config)
    #     save_news_dataset(dataset_train_path, dataset)
    dataset = BaseDataset(config.original_data_path + '/train/behaviors_parsed.tsv',
                        config.original_data_path + '/train/news_parsed.tsv', 
                        config)
    print(f"Load training dataset with size {len(dataset)}.")
    print(time_since(start_time))

    print("load val data:")
    print(time_since(start_time))
    # if os.path.exists(dataset_evaluate_path):
    #     evaluate_dataset = load_news_dataset(dataset_evaluate_path)
    # else:
    #     evaluate_new_path = config.original_data_path + '/val'
    #     evaluate_dataset = NewsDataset(path.join(evaluate_new_path, 'news_parsed.tsv'))
    #     save_news_dataset(dataset_evaluate_path, evaluate_dataset)
    evaluate_new_path = config.original_data_path + '/val'
    evaluate_dataset = NewsDataset(path.join(evaluate_new_path, 'news_parsed.tsv'))

    print("Finish load val data:")
    print(time_since(start_time))

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True))
    print("Finish Build dataloader: ")
    print(time_since(start_time))
    
    if model_name != 'Exp1':
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
    else:
        criterion = nn.NLLLoss()
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            for model in models
        ]
    
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    checkpoint_dir = os.path.join(config.current_data_path + '/checkpoint', config.pretrained_mode, model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # checkpoint_path = latest_checkpoint(checkpoint_dir)
    # if checkpoint_path is not None:
    #     print(f"Load saved parameters in {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path)
    #     early_stopping(checkpoint['early_stop_value'])
    #     step = checkpoint['step']
    #     if model_name != 'Exp1':
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         model.train()
    #     else:
    #         for model in models:
    #             model.load_state_dict(checkpoint['model_state_dict'])
    #             model.train()
    #         for optimizer in optimizers:
    #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    progress = range(1, config.num_epochs * len(dataset) // config.batch_size + 1)
    if should_display_progress():
        progress = tqdm(progress, desc="Training")
    print("Start to do the model training:")
    print(time_since(start_time))

    best_step = 0
    best_checkpoint_path = None
    best_auc_score = float('-inf')

    for i in progress:
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True))
            minibatch = next(dataloader)

        step += 1
        if model_name == 'LSTUR' or model_name == 'LSTURlinear':
            y_pred = model(minibatch["user"], minibatch["clicked_news_length"],
                           minibatch["candidate_news"],
                           minibatch["clicked_news"])
        elif model_name == 'HiFiArk':
            y_pred, regularizer_loss = model(minibatch["candidate_news"],
                                             minibatch["clicked_news"])
        elif model_name == 'TANR':
            y_pred, topic_classification_loss = model(
                minibatch["candidate_news"], minibatch["clicked_news"])
        elif model_name == 'Exp1':
            y_preds = [
                model(minibatch["candidate_news"], minibatch["clicked_news"])
                for model in models
            ]
            y_pred_averaged = torch.stack(
                [F.softmax(y_pred, dim=1) for y_pred in y_preds],
                dim=-1).mean(dim=-1)
            y_pred = torch.log(y_pred_averaged)
        else:
            y_pred = model(minibatch["candidate_news"],
                           minibatch["clicked_news"])

        y = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y)

        if model_name == 'HiFiArk':
            if i % 10 == 0:
                writer.add_scalar('Train/BaseLoss', loss.item(), step)
                writer.add_scalar('Train/RegularizerLoss',
                                  regularizer_loss.item(), step)
                writer.add_scalar('Train/RegularizerBaseRatio',
                                  regularizer_loss.item() / loss.item(), step)
            loss += config.regularizer_loss_weight * regularizer_loss
        elif model_name == 'TANR':
            if i % 10 == 0:
                writer.add_scalar('Train/BaseLoss', loss.item(), step)
                writer.add_scalar('Train/TopicClassificationLoss',
                                  topic_classification_loss.item(), step)
                writer.add_scalar(
                    'Train/TopicBaseRatio',
                    topic_classification_loss.item() / loss.item(), step)
            loss += config.topic_classification_loss_weight * topic_classification_loss
        loss_full.append(loss.item())
        if model_name != 'Exp1':
            optimizer.zero_grad()
        else:
            for optimizer in optimizers:
                optimizer.zero_grad()
        loss.backward()
        if model_name != 'Exp1':
            optimizer.step()
        else:
            for optimizer in optimizers:
                optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), step)
            wandb.log({"Train Loss": loss.item()})

        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )

        if i % config.num_batches_validate == 0:
            (model if model_name != 'Exp1' else models[0]).eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model if model_name != 'Exp1' else models[0], config.original_data_path + '/val',
                config.num_workers, evaluate_dataset, 200000)
            (model if model_name != 'Exp1' else models[0]).train()
            writer.add_scalar('Validation/AUC', val_auc, step)
            writer.add_scalar('Validation/MRR', val_mrr, step)
            writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
            writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
            tqdm.write(
                f"\nTime {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
            )
            wandb.log({"validation AUC": val_auc, "validation MRR": val_mrr, "validation nDCG@5":val_ndcg5, "validation nDCG@10":val_ndcg10})

            early_stop, get_better = early_stopping(-val_auc)
            if early_stop:
                tqdm.write('Early stop.')
                break
            elif get_better:
                try:
                    best_step = step
                    if best_checkpoint_path:
                        os.remove(best_checkpoint_path)
                    best_checkpoint_path = config.current_data_path + f"/checkpoint/{config.pretrained_mode}/{model_name}/ckpt-{step}.pth"

                    torch.save(
                        {
                            'model_state_dict': (model if model_name != 'Exp1'
                                                 else models[0]).state_dict(),
                            'optimizer_state_dict':
                            (optimizer if model_name != 'Exp1' else
                             optimizers[0]).state_dict(),
                            'step':
                            step,
                            'early_stop_value':
                            -val_auc
                        }, best_checkpoint_path)
                except OSError as error:
                    print(f"OS error: {error}")

    # load and print best
    best_checkpoint_path = config.current_data_path + f"/checkpoint/{config.pretrained_mode}/{model_name}/ckpt-{best_step}.pth"
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params, trainable_params = count_parameters(model)
    print("total_params, trainable_params is:")
    print(total_params)
    print(trainable_params)
    
    best_auc, best_mrr, best_ndcg5, best_ndcg10 = evaluate(model, config.original_data_path + '/val', config.num_workers, news_dataset_built=evaluate_dataset)

    # evaluate_popular(model, config.original_data_path + '/val', config.num_workers, news_dataset_built=evaluate_dataset)
    tqdm.write(
                f"\n\nTime {time_since(start_time)}, \nbatches {i}, \nFinal AUC: {best_auc:.4f}, \nFinal MRR: {best_mrr:.4f}, \nFinal nDCG@5: {best_ndcg5:.4f}, \nFinal nDCG@10: {best_ndcg10:.4f}, "
            )
    wandb.log({"Final AUC": best_auc, "Final MRR": best_mrr, "Final nDCG@5":best_ndcg5, "Final nDCG@10":best_ndcg10})

    wandb.finish()


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    # if config.using_UNK==0:
    #     with open(config.attack_folder + "/dataset_config_yaml.json", "r") as f:
    #         dataset_config = json.load(f)
    #     for key, value in dataset_config.items():
    #         setattr(config, key, value)
    print('Using device:', device)
    print(f'Training model {model_name}')
    # print(config.num_words)
    # print(config.num_categories)
    train()
