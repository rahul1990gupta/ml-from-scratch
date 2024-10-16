import numpy as np
import pandas as pd 


class TrainSet:
    def __init__(self, df):
        cols = df.columns
        self.global_mean = df[cols[2]].mean() 

        df.columns = ["user_id", "item_id", "rating"]
        self.df = df
        self.populate()

    def populate(self):
        self.n_items = self.df["item_id"].nunique()
        self.n_users = self.df["user_id"].nunique()

        self.sorted_user_ids = sorted(set(self.df["user_id"].tolist()))
        self.raw_to_internal_user_id = {
            user_id: i for i, user_id in enumerate(self.sorted_user_ids)
        }

        self.sorted_item_ids = sorted(set(self.df["item_id"].tolist()))
        self.raw_to_internal_item_id = {
            item_id: i for i, item_id in enumerate(self.sorted_item_ids)
        }
    
    def knows_user(self, u):
        return u in self.sorted_user_ids
    
    def knows_item(self, i):
        return i in self.sorted_item_ids

    def get_all_ratings(self):
        records = self.df.to_records(index=False)
        records_with_internal_ids = [
            (self.raw_to_internal_user_id[u], self.raw_to_internal_item_id[i], rating)
            for u, i, rating in records
        ]
        return records_with_internal_ids

class SVD:
    def __init__(self, verbose=False):
        self.n_factors = 100
        self.n_epochs = 20 
        self.init_mean = 0
        self.init_std_dev = .1 
        self.lr_all = .005
        self.reg_all = .02
        self.verbose = verbose

    def fit(self, data: TrainSet):
        self.trainset = data

        bu = np.zeros(data.n_users, dtype=np.double)
        bi = np.zeros(data.n_items, dtype=np.double)
        pu = np.random.normal(self.init_mean, self.init_std_dev, size=(data.n_users, self.n_factors))
        qi = np.random.normal(self.init_mean, self.init_std_dev, size=(data.n_items, self.n_factors))


        global_mean = data.global_mean

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            for u, i, r in data.get_all_ratings():
                # print(u, i, r)
                # import pdb;pdb.set_trace()
                dot = np.dot(qi[i].T, pu[u])
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                bu[u] += self.lr_all * (err - self.reg_all * bu[u])
                bi[i] += self.lr_all * (err - self.reg_all * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += self.lr_all * (err * qif - self.reg_all * puf)
                    qi[i, f] += self.lr_all * (err * puf - self.reg_all * qif)


        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)


    def predict(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        est = self.trainset.global_mean

        if known_user:
            est += self.bu[u]

        if known_item:
            est += self.bi[i]

        if known_user and known_item:
            est += np.dot(self.qi[i], self.pu[u])


def get_algo(df):
    data = TrainSet(df)
    algo = SVD()
    algo.fit(data)
    return algo


if __name__== "__main__":
    data = {
        "userid": [i//3 + 1 for i in range(9)],
        "itemid": [ i%3 + 1 for i in range(9)],
        "ratings": [i  for i in range(9)]
    }
    df = pd.DataFrame.from_dict(data)
    ts = TrainSet(df)
    print(ts.ratings)
