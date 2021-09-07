
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(160, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 17384)
        self.softmax5 = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax5(x)

        return x


model = Net()
model.to(device)

user_watch=np.load('user_watch.npy')
user_search=np.load('user_search.npy')
user_feat=np.load('user_feat.npy')
user_labels= np.load('user_labels.npy')
inputs = np.hstack((user_watch, user_search, user_feat))
x_data = torch.FloatTensor(inputs)
y_data = torch.FloatTensor(user_labels)

deal_dataset = TensorDataset(x_data, y_data)
train_size = int(0.91 * len(deal_dataset))
test_size = len(deal_dataset) - train_size
train_data_set, test_data_set = torch.utils.data.random_split(deal_dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_data_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data_set, batch_size=32, shuffle=True)


criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01, )


def train(x, target, model):
    opt.zero_grad()
    pred = model(x)
    temp = target.reshape(-1, pred.shape[0])[0].long()
    loss = criterion(pred, temp)
    loss.backward()
    opt.step()
    return loss.detach().item()


def train_acc(train_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for input, label in train_loader:
            input=input.to(device)
            label=label.to(device)
            outputs = model(input)
            value, indicies = torch.topk(outputs, 10, dim=1)
            total += label.size(0)
            correct += topK(label, indicies)
    print("Accuracy {:.2f}%".format(100 * correct / total))
    # file = open("res.txt", 'a+')
    # file.write("{:.2f}%\n".format(100. * correct / total))
    # file.flush()
    # file.close()


def topK(labels, indicy):
    upper = labels.size(0)
    labels = labels.numpy()
    indicy = indicy.numpy()
    hit = 0
    for i in range(upper):
        for h in range(10):
            if indicy[i][h] == labels[i][0]:
                hit += 1
                break
    return hit


if __name__ == "__main__":
    for i in range(20):
        running_loss = 0
        model.train()
        for images, labels in train_loader:
            loss = train(images, labels, model)
            running_loss += loss
        print("Epoch {} - Training loss:{}".format(i, running_loss / len(train_loader)))
        train_acc(train_loader)
        train_acc(test_loader)