
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn


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
            X = torch.zeros((self.n_lines, self.feature_size), dtype=torch.float)
            Y = torch.zeros((self.n_lines), dtype=torch.int8)
            qid = torch.zeros((self.n_lines), dtype=torch.int32)
    
            for ix, line in enumerate(fp):
                if ix % 10000 ==9990:
                    print("Read lines:", ix)
                    # break


                tokens = line.strip().split(" ")
                Y[ix] = int(tokens[0])
                # print(tokens)
                features = [token.split(":")[1] for token in tokens[2:]]
                # print(features)
                X[ix] = torch.Tensor([float(f) for f in features])
                qid[ix] = int(tokens[1].split(":")[1])

        self.X, self.Y = X, Y

    def __getitem__(self, idx):
        y = torch.zeros((5))
        y[self.Y[idx]] = 1
        return self.X[idx], y

    def __len__(self):
        return self.n_lines


# Test accuracy: 
class PointWise(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(136, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        # probs = torch.softmax(logits, dim=1)
        return logits

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def run_nn():
    train_data = MSLR10k("train") 
    test_data = MSLR10k("test")
    # sample_data = MSLR10k("sample")

    batch_size = 64

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    # sample_dataloader = DataLoader(sample_data, batch_size=batch_size)

    model = PointWise().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        # train(sample_dataloader, model, loss_fn, optimizer)
    print("Done!")

run_nn()

##########################################################################


# download the data from the link below
# https://www.microsoft.com/en-us/research/project/mslr/
# https://github.com/yanshanjing/RankNet-Pytorch/blob/master/RankNet-Pytorch.py

class RankNet:
    pass 


# https://github.com/airalcorn2/RankNet/blob/master/lambdarank.py
class Lambdarank:
    pass 


