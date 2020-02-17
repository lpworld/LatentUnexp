from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
import json
import pandas as pd

alpha = 0.1
data = pd.read_csv('test.csv')
business_id = list(set(data['business_id']))
with open('unexpectedness.json','r') as f:
    unexp = json.load(f)
with open('relevance.json','r') as f:
    relevance = json.load(f)
with open('primitive.json','r') as f:
    primitive = json.load(f)

def rmse(predictions, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')
    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)
    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))
    return rmse_


def mae(predictions, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')
    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])
    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))
    return mae_

def rmse_lc(predictions, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')
    mse = np.mean([float((true_r - est - alpha*unexp[uid][bid])**2)
                   for (uid, bid, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)
    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))
    return rmse_


def mae_lc(predictions, verbose=True):
    if not predictions:
        raise ValueError('Prediction list is empty.')
    mae_ = np.mean([float(abs(true_r - est - alpha*unexp[uid][bid]))
                    for (uid, bid, true_r, est, _) in predictions])
    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))
    return mae_

def precision(predictions, k=10, verbose=True, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    return np.mean(list(precisions.values()))

def precision_lc(predictions, k=10, verbose=True, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, bid, true_r, est, _ in predictions:
        distance = est + alpha*unexp[uid][bid]
        user_est_true[uid].append((distance, true_r))
    precisions = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    return np.mean(list(precisions.values()))

def recall(predictions, k=10, verbose=True, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return np.mean(list(recalls.values()))

def recall_lc(predictions, k=10, verbose=True, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, bid, true_r, est, _ in predictions:
        distance = est + alpha*unexp[uid][bid] 
        user_est_true[uid].append((distance, true_r))
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return np.mean(list(recalls.values()))

def unexpectedness(predictions, k=5, verbose=True):
    user_est_true = defaultdict(list)
    score = []
    for uid, bid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, bid))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        for bid in [x[1] for x in user_ratings[:k]]:
            distance = unexp[uid][bid]
            score.append(distance)
    score = np.mean(score)
    if verbose:
        print('UNEXPECTEDNESS:  {0:1.4f}'.format(score))
    return score

def unexpectedness_lc(predictions, k=5, verbose=True):
    user_est_true = defaultdict(list)
    score = []
    for uid, bid, true_r, est, _ in predictions:
        distance = est + alpha*unexp[uid][bid]
        user_est_true[uid].append((distance, bid))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        for bid in [x[1] for x in user_ratings[:k]]:
            distance = unexp[uid][bid]
            score.append(distance)
    score = np.mean(score) 
    if verbose:
        print('UNEXPECTEDNESS:  {0:1.4f}'.format(score))
    return score

def coverage(predictions, k=5, verbose=True):
    data = pd.read_csv('test.csv')
    user_est_true = defaultdict(list)
    business = []
    for uid, bid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, bid))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        business.append([x[1] for x in user_ratings[:k]])
    business = list(set([y for x in business for y in x]))
    score = len(business) / len(set(data['business_id']))
    if verbose:
        print('COVERAGE:  {0:1.4f}'.format(score))
    return score

def coverage_lc(predictions, k=5, verbose=True):
    data = pd.read_csv('test.csv')
    user_est_true = defaultdict(list)
    business = []
    for uid, bid, true_r, est, _ in predictions:
        distance = est + alpha*unexp[uid][bid]
        user_est_true[uid].append((distance, bid))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        business.append([x[1] for x in user_ratings[:k]])
    business = list(set([y for x in business for y in x]))
    score = len(business) / len(set(data['business_id']))
    if verbose:
        print('COVERAGE:  {0:1.4f}'.format(score))
    return score

def serendipity(predictions, k=5, verbose=True):
    user_est_true = defaultdict(list)
    score = []
    for uid, bid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, bid))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        bid = [x[1] for x in user_ratings[:k] if x[1] in primitive[uid]]
        score.append(len(bid) / k)
    score = 1 - np.mean(score)
    if verbose:
        print('SERENDIPITY:  {0:1.4f}'.format(score))
    return score

def serendipity_lc(predictions, k=5, verbose=True):
    user_est_true = defaultdict(list)
    score = []
    for uid, bid, true_r, est, _ in predictions:
        distance = est + alpha*unexp[uid][bid]
        user_est_true[uid].append((distance, bid))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        bid = [x[1] for x in user_ratings[:k] if x[1] in primitive[uid]]
        score.append(len(bid) / k)
    score = 1 - np.mean(score)
    if verbose:
        print('SERENDIPITY:  {0:1.4f}'.format(score))
    return score

def diversity(predictions, k=5, verbose=True):
    user_est_true = defaultdict(list)
    score = []
    for uid, bid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, bid, true_r))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        bid = [x[1] for x in user_ratings[:k] if x[1] not in primitive[uid] and x[2] > 3.5]
        score.append(len(bid) / k)
    score = np.mean(score)
    if verbose:
        print('DIVERSITY:  {0:1.4f}'.format(score))
    return score

def diversity_lc(predictions, k=5, verbose=True):
    user_est_true = defaultdict(list)
    score = []
    for uid, bid, true_r, est, _ in predictions:
        distance = est + alpha*unexp[uid][bid]
        user_est_true[uid].append((distance, bid, true_r))
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        bid = [x[1] for x in user_ratings[:k] if x[1] not in primitive[uid] and x[2] > 3.5]
        score.append(len(bid) / k)
    score = np.mean(score)
    if verbose:
        print('DIVERSITY:  {0:1.4f}'.format(score))
    return score