# Overview 
Search ranking is an omni-prsent problem that we have faced since the beginning of the internet. In this exercise, we will implement stages of solutions that can be used to solve a search ranking problem.


## Classical search engine with fake dataset. 
- implement spell checker
- implement tfidf 
- implement PageRank 

These techniques are also used as features to power the Machine Learning model we will encounter in the next section. 

## Leanring to rank with mslr dataset
- implement NDCG
- implement Tree based models for prediction. 
- implement a basic feed-forward neural network.

Even though Neural Network are all the rage in computer vision and text related problem. Here, classical tree based model do very well compatred to neural network. As you have verified by compring the performance of gradient boposted tree with a neural network 

Some popular Neural Networks:
- for pairwise ranking algorithm: RankNet/RankBoost 
- for listwise ranking algorithm LambdaRank/ LambdaMART

You can find more algorithms on https://en.wikipedia.org/wiki/Learning_to_rank

# Appendix 
Metric: NDCG 
Flow: spell checker -> query expansion -> query understanding -> document selection -> ranker -> blender 

Ranking 
- logistic regression 
- lambda Mart 
- lambda Rank 


