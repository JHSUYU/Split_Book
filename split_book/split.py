from torch import nn, optim
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from splitnn_net import SplitNN
import numpy as np
from sklearn import metrics
from distribute_data import Distribute_Bookcrossing


input_size = [160, 160]
hidden_sizes = {"client_1": [128], "client_2": [128], "server": [128, 128]}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

watch_vec_size = 64
search_vec_size = 64
other_feat_size = 32
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



def train(x, target, splitNN):
    target.to(device)
    splitNN.zero_grads()
    pred = splitNN.forward(x).to(device)
    criterion = nn.CrossEntropyLoss()
    temp = target.reshape(-1, pred.shape[0])[0].long().to(device)
    loss = criterion(pred, temp)
    loss.backward()
    splitNN.step()
    return loss.detach().item()


def train_acc(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs = splitnn.forward(data_ptr)
            value, indicies = torch.topk(outputs, 10, dim=1)
            total += label.size(0)
            correct += topK(label, indicies)
    print("Accuracy {:.2f}%".format(100 * correct / total))


def topK(labels, indicy):
    upper = labels.size(0)
    labels = labels.cpu().numpy()
    indicy = indicy.cpu().numpy()
    hit = 0
    for i in range(upper):
        for h in range(10):
            if indicy[i][h] == labels[i][0]:
                hit += 1
                break
    return hit


def print_parameters(model):
    res = {}
    for name, param in models["server"].named_parameters():
        res[name] = param.data.numpy()
        grad_key = name+"_grad"
        res[grad_key] = param.grad.data.numpy()


def cal_activation_num(splitNN,distributed_dataset,data_owners):
    count=[0,0,0,0]
    count_num=0
    for images,labels in distributed_dataset:
        count_num+=1
        splitNN.zero_grads()
        client_output = {}
        remote_outputs = []
        i = 0
        for owner in data_owners:
            if i == 0:
                input = images[owner].reshape([-1, 160])
                for j in range(len(splitNN.models[owner])):
                    input = splitNN.models[owner][j](input)
                    if j==1:
                        for t in range(len(input.detach().numpy().tolist())):
                            for data in input.data.numpy().tolist()[t]:
                                if data > 0.0 :
                                    count[0] += 1
                client_output[owner] = input
                remote_outputs.append(client_output[owner].requires_grad_())
                i += 1
            else:
                input = images[owner].reshape([-1, 160])
                for j in range(len(splitNN.models[owner])):
                    input = splitNN.models[owner][j](input)
                    if (j == 1):
                        for t in range(len(input.detach().numpy().tolist())):
                            for data in input.data.numpy().tolist()[t]:
                                if data > 0.0:
                                    count[1] += 1

                client_output[owner] = input
                remote_outputs.append(client_output[owner].requires_grad_())
        server_input = torch.cat(remote_outputs, 1)
        for j in range(len(splitNN.models["server"])):
            server_input = splitNN.models["server"][j](server_input)
            if  (j == 1 or j == 3):
                for t in range(len(server_input.detach().numpy().tolist())):
                    for data in server_input.data.numpy().tolist()[t]:
                        if (data > 0.0):
                            if (j == 1):
                                count[2] += 1
                            if (j == 3):
                                count[3] += 1
        server_output = server_input
    count[0] /= (128 * 600)
    count[1] /= (128 * 600)
    count[2] /= (128 * 600)
    count[3] /= (64 * 600)
    print(count)
    print(count_num)

# def importance_estimation(model,name_model):
#     res=dict()
#     parameters_for_update = []
#     parameters_for_update_named = []
#     for name, param in model.named_parameters():
#         parameters_for_update.append(param)
#         parameters_for_update_named.append((name, param))
#     pruning_parameters_list = list()
#     for module_index, m in enumerate(model.modules()):
#         print(module_index)
#         print(m)
#         if module_index % 2 == 1:
#             m_to_add = m
#             for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
#                            "compute_criteria_from": m_to_add.weight}
#             pruning_parameters_list.append(for_pruning)
#     pruning_engine = pytorch_pruning(pruning_parameters_list)
#     print("------" + name_model + "importance------")
#     for i in range(len(pruning_engine.parameters)):
#         print("------layer"+str(i)+"------")
#         nunits = pruning_engine.parameters[i].size(0)
#         criteria_for_layer = (pruning_engine.parameters[i] * pruning_engine.parameters[i].grad).data.pow(2).view(nunits,-1).sum(dim=1)
#         key="layer"+str(i)
#         res[key]=criteria_for_layer
#     return res

def cal_auc(dataloader):
    score=[]
    labels=[]
    label_res=[]

    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs=splitnn.forward(data_ptr)
            for output in outputs.tolist():
                score.append(output)
            for i in label.tolist():
                label_res.append(int(i[0]))
                temp = []
                for j in range(3952):
                    temp.append(0)
                temp[int(i[0])]=1
                labels.append(temp)
    score=np.array(score)
    labels=np.array(labels)
    print(metrics.roc_auc_score(labels, score, average='micro'))
    return




def data_test_acc(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data_ptr, label in dataloader:
            outputs = splitnn.forward(data_ptr)
            value, indicies = torch.topk(outputs, 10, dim=1)
            total += label.size(0)
            correct += topK(label, indicies)
    print("Accuracy {:.2f}%".format(100 * correct / total))
    return (100*correct/total)



def train_model(epoch, dataset):
    res=[]
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
    distributed_trainloader = Distribute_Bookcrossing(data_owners=data_owners, data_loader=train_loader)
    distributed_testloader = Distribute_Bookcrossing(data_owners=data_owners, data_loader=test_loader)
    for i in range(epoch):
        running_loss = 0
        splitnn.train()
        for images, labels in distributed_trainloader:
            loss = train(images, labels, splitnn)
            running_loss += loss
        print("Epoch {} - Training loss:{}".format(i, running_loss / len(train_loader)))
        train_acc(distributed_trainloader)
        temp=data_test_acc(distributed_testloader)
        res.append(temp)
        #cal_auc(distributed_testloader)
        # importance_estimation(splitnn.models['client_1'], "client_1")
        # importance_estimation(splitnn.models['client_2'], "client_2")
        # importance_estimation(splitnn.models['server'], "server")
    return max(res)

if __name__=="__main__":
    res=[]
    for j in range(30):
        models = {
            "client_1": nn.Sequential(
                nn.Linear(input_size[0], 128),
                nn.ReLU(),
            ),

            "client_2": nn.Sequential(
                nn.Linear(input_size[1], 128),
                nn.ReLU(),
            ),

            "server": nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 17384),
                nn.LogSoftmax(dim=1)
            )
        }
        deal_dataset = TensorDataset(x_data, y_data)
        data_owners = ['client_1', 'client_2']
        model_locations = ['client_1', 'client_2', 'server']
        for location in model_locations:
            for param in models[location]:
                param.to(device)
        optimizers = [optim.Adam(models[location].parameters(), lr=0.01, ) for location in model_locations]
        splitnn = SplitNN(models, optimizers, data_owners).to(device)
        temp=train_model(18,deal_dataset)
        res.append(temp)
    sum=0
    for i in res:
        sum+=i
    print("Accuracy {:.2f}%".format( sum / len(res)))
    # importance_estimation(splitnn.models['client_1'], "client_1")
    # importance_estimation(splitnn.models['client_2'], "client_2")
    # importance_estimation(splitnn.models['server'], "server")













