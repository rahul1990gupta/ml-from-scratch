from collections import defaultdict, Counter
import math

animals = []
with open("data/animals.txt") as f:
    for line in f.readlines():
        animals += [word.strip().lower() for word in line.split("\t")]

def spell_correct(word):
    # We only handle one letter replace 
    letters = "abcdefghijklmnopqrstuvwxyz"
    all_replaced_words = []
    for i in range(0, len(word)):
        for c in letters:
            replaced_word = word[:i] + c + word[i+1:]

            if replaced_word in animals:
                return replaced_word
    return word

from nltk.stem import WordNetLemmatizer

def build_index(documents):
    l = WordNetLemmatizer()
    iix = defaultdict(list)
    tf = {}
    for i, doc in enumerate(documents):
        tf[i] = defaultdict(int)
        words = [word.strip().lower() for word in  doc.split(" ")]
        words = [l.lemmatize(word, pos="n") for word in words]
        c = Counter(words)
        for word, count in c.items():
            iix[word].append(i)
            tf[i][word] += count
    
    return iix, tf


def tfidf(q, iix, tf, documents):
    """
    This is implemented for a single word query
    """
    # This implements count-idf
    tfidf_score = [0 for _ in documents]
    
    idf = math.log(len(documents)/ len(iix[q]))
    for i, doc in enumerate(documents):
        raw_tf = tf[i][q]
        tfidf_score[i] = raw_tf/idf

    return [i for i, score in enumerate(tfidf_score) if score> 0]          


def pagerank(n_nodes, edges):
    
    in_bound = defaultdict(list)
    out_bound = defaultdict(list)

    for u, v in edges:
        out_bound[u].append(v)
        in_bound[v].append(u)
    


    n_epoch = 10
    damp = 0.85
    pr_scores_prev = [1./n_nodes for _ in range(n_nodes)]
    pr_scores_next = [0 for _ in range(n_nodes)]

    for i in range(n_epoch):
        # u -> v
        for v in range(n_nodes):
            pr_score = 0
            for u in in_bound[v]:
                pr_score += pr_scores_prev[u]/len(out_bound[u])
            
            pr_scores_next[v] = (1-damp)/n_nodes +  damp*pr_score
        pr_scores_prev = pr_scores_next

    return pr_scores_prev
