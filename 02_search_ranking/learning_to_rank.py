import numpy as np 
from collections import Counter 

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


from torch.utils.data import Dataset

class MSLR10k(Dataset):
    def __init__(self, dtype):
        self.fname = f"MSLR-WEB10K/Fold1/{dtype}.txt"
        self.feature_size = 136
        self.count_lines()
        self.read()

    def count_lines(self):
        n_lines = 0
        with open(self.fname) as fp:
            for line in fp: 
                n_lines += 1
        self.n_lines = n_lines

    def read(self):
        with open(self.fname, "r") as fp:
            X = np.zeros((self.n_lines, self.feature_size), dtype=np.float64)
            Y = np.zeros((self.n_lines), dtype=np.int8)
            qid = np.zeros((self.n_lines), dtype=np.int32)
    
            for ix, line in enumerate(fp):
                if ix % 10000 ==9990:
                    print("Read lines:", ix)
                    # break


                tokens = line.strip().split(" ")
                Y[ix] = int(tokens[0])
                # print(tokens)
                features = [token.split(":")[1] for token in tokens[2:]]
                X[ix] = np.asfarray(features)
                qid[ix] = tokens[1].split(":")[1]

        self.X, self.Y = X, Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.n_lines


# Base line accuracy : 52% 
def baseline():
    train_data = MSLR10k("train")
    pred = np.zeros((train_data.n_lines), dtype=np.int8)
    accuracy = np.sum(pred == train_data.Y)/ train_data.n_lines

    print(Counter(train_data.Y.T[0]).items())
    print("Accuracy", accuracy) 
    


from sklearn.tree import DecisionTreeClassifier

# Decision Tree: 55 % 
def train_decision_tree():
    train_data = MSLR10k("train")

    clf = DecisionTreeClassifier(random_state=0, max_depth=4)
    clf.fit(train_data.X, train_data.Y)

    pred = clf.predict(train_data.X)

    accuracy = np.sum(pred == train_data.Y)/ train_data.n_lines
    print("Accuracy", accuracy) 


from sklearn.ensemble import RandomForestClassifier

train_data = MSLR10k("train")

# Accuracy: 55 %
def train_random_forest():
    clf = RandomForestClassifier(n_estimators=20, max_depth=4)
    clf.fit(train_data.X, train_data.Y)
    
    pred = clf.predict(train_data.X)

    accuracy = np.sum(pred == train_data.Y)/ train_data.n_lines
    print("Accuracy of random forest model", accuracy) 


from sklearn.ensemble import HistGradientBoostingClassifier

# Accuracy: 58%
def train_gradient_boosted_trees():
    clf = HistGradientBoostingClassifier(max_depth=4)
    clf.fit(train_data.X, train_data.Y)

    pred = clf.predict(train_data.X)

    accuracy = np.sum(pred == train_data.Y)/ train_data.n_lines
    print("Accuracy of gradient boosted trees", accuracy) 


train_decision_tree()
train_gradient_boosted_trees()
