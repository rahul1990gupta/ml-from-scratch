import numpy as np 

def dcg(relevance_scores, k):
    relevance_scores = np.asfarray(relevance_scores)[:k]
    if relevance_scores.size == 0:
        return 0.0
    numerator = 2 ** relevance_scores - 1
    denominator = np.log2(np.arange(1, relevance_scores.size + 1) + 1)
    return np.sum( numerator / denominator)


def ndcg(relevance_scores, k):
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg

# https://www.microsoft.com/en-us/research/project/mslr/
# https://github.com/yanshanjing/RankNet-Pytorch/blob/master/RankNet-Pytorch.py

class RankNet:
    pass 


# https://github.com/airalcorn2/RankNet/blob/master/lambdarank.py
class Lambdarank:
    pass 
