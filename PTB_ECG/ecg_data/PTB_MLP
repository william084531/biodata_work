import numpy as np
from csv import reader
from pandas.core.frame import DataFrame
import pandas as pd
from itertools import chain
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable

train_d = pd.read_csv('Drive/ecg_competition/train_ecg.csv')
train_d = train_d.set_index('index_label')
train_t = np.load('Drive/ecg_competition/train_label_ecg.npy')
test_d = pd.read_csv('Drive/ecg_competition/test_ecg.csv')
test_d = test_d.set_index('index_label')
test_t = np.load('Drive/ecg_competition/test_label_ecg.npy')
traintarget = torch.from_numpy(train_t.reshape(len(train_t))).type(torch.LongTensor)
testtarget = torch.from_numpy(test_t.reshape(len(test_t))).type(torch.LongTensor)
inputdata = torch.from_numpy(train_d.values.astype(float)).type(torch.FloatTensor)
testdata = torch.from_numpy(test_d.values.astype(float)).type(torch.FloatTensor)
#validation = torch.from_numpy(validation.values.astype(float).reshape(1413,1,1,8)).type(torch.FloatTensor)
train_data = Data.TensorDataset(inputdata,traintarget)
test_data = Data.TensorDataset(testdata,testtarget)
train_loader = Data.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=256, shuffle=True)
#%%
class MLP(nn.Module):
    def __init__(self):
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(700, 1000)
      self.relu1 = nn.ReLU()
      self.fc2 = nn.Linear(1000, 1600)
      self.relu2 = nn.ReLU()
      self.fc3 = nn.Linear(1600, 1600)
      self.relu3 = nn.ReLU()
      self.fc4 = nn.Linear(1600, 1600)
      self.relu4 = nn.ReLU()
      self.fc5 = nn.Linear(1600, 1600)
      self.relu5 = nn.ReLU()
      self.fc6 = nn.Linear(1600, 1000)
      self.relu6 = nn.ReLU()
      self.fc7 = nn.Linear(1000, 100)
      self.relu7 = nn.ReLU()
      self.fc8 = nn.Linear(100, 2)
      
      
    def forward(self, x):
      out = self.fc1(x)
      out = self.relu1(out)
      out = self.fc2(out)
      out = self.relu2(out)
      out = self.fc3(out)
      out = self.relu3(out)
      out = self.fc4(out)
      out = self.relu4(out)
      out = self.fc5(out)
      out = self.relu5(out)
      out = self.fc6(out)
      out = self.relu6(out)
      out = self.fc7(out)
      out = self.relu7(out)
      out = self.fc8(out)
      return out

#%%
torch.manual_seed(1) # 表示torch tensor 一開始的隨機值，如此一來下次在進行訓練時，隨機出來的tenspr就會是相同的!!
#num_cores = 8 #可以試試看其他的CPU 核心 number of cores表示所使用的核心處理器是第幾核
lr = 0.001
#devices = [':{}'.format(n) for n in range(0, num_cores)]
mlp = MLP()
use_cuda = True
if use_cuda and torch.cuda.is_available():
    mlp.cuda()
model_save_name = 'params_mlp_2019_9_13.pkl'
path = F"/content/gdrive/My Drive/{model_save_name}" 
mlp.load_state_dict(torch.load(path))
loss_func = nn.CrossEntropyLoss() # 已經內涵softmax所以不需要另外加
#cnn.load_state_dict(torch.load('params_cnn_2019_06_22_1_3.pkl'))
optimizer = torch.optim.Adam(mlp.parameters(), lr = lr)   # optimize all cnn parameters(控制模型學習率的變化)


#%%
acc = []
losses = []
acces = []
eval_losses = []
eval_acces = []
ys_mlp = []
yl_mlp = []
pre = []
lab = []
#def adjust_learning_rate(optimizer, lr):
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr
#    adjust_learning_rate(optimizer, epoch)
for e in range(2):
    train_loss = 0
    train_acc = 0
    mlp.train()
    for im, label in train_loader:
#        if e > 30:
#          lr = 0.00001
#          adjust_learning_rate(optimizer, lr)
#        print(optimizer)
        if use_cuda and torch.cuda.is_available():
          im = Variable(im).cuda()
          label = Variable(label).cuda()
        # 前向传播
        out = mlp(im)
        loss = loss_func(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
#    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    mlp.eval() # 将模型改为预测模式
    for im, label in test_loader:
        if use_cuda and torch.cuda.is_available():
          im = Variable(im).cuda()
          label = Variable(label).cuda()
        out = mlp(im)
        loss = loss_func(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        ys_mlp.append(out.cpu().data.numpy())
        yl_mlp.append(label.cpu().numpy())
        pre.append(pred)
        lab.append(label)
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_loader), train_acc / len(train_loader), 
                     eval_loss / len(test_loader), eval_acc / len(test_loader)))
    model_save_name = 'params_mlp_2019_9_13.pkl'
    path = F"/content/gdrive/My Drive/{model_save_name}" 
    torch.save(mlp.state_dict(), path)
