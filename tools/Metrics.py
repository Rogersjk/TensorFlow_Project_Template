"""
@author: Junkai Sun
@file: Metrics.py
@time: 2018/4/24 17:22
"""
# refer to https://www.kaggle.com/davidgasquez/ndcg-scorer?scriptVersionId=140721
import numpy as np
import scipy.stats as stats

def kendall_coefficient(y_true, y_pred):
    num = len(y_true)
    conc, disc = 0, 0
    for i in range(num):
        for j in range(i+1, num):
            if (y_true[i]-y_true[j]) * (y_pred[i]-y_pred[j]) < 0:
                disc += 1
            elif (y_true[i]-y_true[j]) * (y_pred[i]-y_pred[j]) > 0:
                conc += 1
    return (conc - disc) / (num * (num-1) / 2)

def dcg_score(real_relevance, y_pred, k=10):
    # refer to https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    order = np.argsort(y_pred)[::-1]
    # rel = np.take(y_true, order[:k])
    rel = np.take(real_relevance, order[:k])
    # rel = np.exp2(np.take(y_true, order[:k]) / k) - 1
    # dcg = np.sum(rel / np.log2(np.arange(2, len(rel)+2)))
    dcg = np.sum((np.exp2(rel)-1) / np.log2(np.arange(2, len(rel)+2)))
    return dcg


def ndcg_score(y_true, y_pred, k=10):
    '''
     y_true:
        ground truth of each sample
     y_pred:
        prediction for each sample
     k:
        select top k
    return:
        NDCG @ k score
    '''
    order = np.argsort(y_true)
    # transform real value to relevance according to the sort order
    real_relevance = np.zeros_like(y_true)
    real_relevance[order] = np.arange(len(y_true)) / len(y_true)
    # y_true[order] = np.arange(len(y_true)) / len(y_true)
    ideal_dcg = dcg_score(real_relevance, y_true, k)
    actual_dcg = dcg_score(real_relevance, y_pred, k)
    return actual_dcg / ideal_dcg

def nsdk(y_true, y_pred, k=10):
    pass

if __name__ == '__main__':
    # print(ndcg_score(np.array([5, 3, 2]), np.array([5, 3, 4])))
    # print(dcg_score([4, 3, 2], [2, 1, 0]))
    print(ndcg_score(np.arange(0, 20.0), np.arange(20.0, 0, -1), 10))
    print(ndcg_score(np.arange(0, 20.0), np.arange(0, 20, 1)))
    print(ndcg_score(np.arange(0, 20.0), np.arange(0, 20, 1)))

    # Average precision
    x1 = [12, 2, 1, 12, 2]
    x2 = [1, 4, 7, 1, 0]
    print(kendall_coefficient(x1, x2))

