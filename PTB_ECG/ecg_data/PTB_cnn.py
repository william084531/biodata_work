import numpy as np
from csv import reader
from pandas.core.frame import DataFrame
import pandas as pd
from itertools import chain
import torch
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
#load 資料進入
train_d = pd.read_csv('Drive/ecg_competition/train_ecg.csv')
train_d = train_d.set_index('index_label')
train_t = np.load('Drive/ecg_competition/train_label_ecg.npy')
test_d = pd.read_csv('Drive/ecg_competition/test_ecg.csv')
test_d = test_d.set_index('index_label')
test_t = np.load('Drive/ecg_competition/test_label_ecg.npy')
traintarget = torch.from_numpy(train_t.reshape(len(train_t))).type(torch.LongTensor)
testtarget = torch.from_numpy(test_t.reshape(len(test_t))).type(torch.LongTensor)
inputdata = torch.from_numpy(train_d.values.astype(float).reshape(len(train_d),1,700)).type(torch.FloatTensor)
testdata = torch.from_numpy(test_d.values.astype(float).reshape(len(test_d),1,700)).type(torch.FloatTensor)
train_data = Data.TensorDataset(inputdata,traintarget)
test_data = Data.TensorDataset(testdata,testtarget)
train_loader = Data.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=256, shuffle=True)
#%%
class CNN(nn.Module):
    def __init__(self):
      super(CNN, self).__init__()
      self.conv1 = nn.Sequential(
        nn.Conv1d(
          in_channels=1,
          out_channels=100,
          kernel_size= 5),
        nn.ReLU()
        )
      self.conv2 = nn.Sequential(
        nn.Conv1d(100,100,10),
        nn.ReLU(),
        nn.MaxPool1d(3)
        )
      self.conv3 = nn.Sequential(
        nn.Conv1d(100,100,10),
        nn.ReLU()
        )
      self.conv4 = nn.Sequential(
        nn.Conv1d(100,160,10),
        nn.ReLU(),
        nn.AvgPool1d(10),
        nn.Dropout(0.5)
        )
      self.fc1 = nn.Linear(3360, 100)
      self.activation1 = nn.ReLU()
      self.fc2 = nn.Linear(100, 2)
      
      
    def forward(self, x):
      out = self.conv1(x)
      out = self.conv2(out)
      out = self.conv3(out)
      out = self.conv4(out)
      out = out.view(out.size(0), -1)
      out = self.fc1(out)
      out = self.activation1(out)
      out = self.fc2(out)
      return out

#%%
torch.manual_seed(1) # 表示torch tensor 一開始的隨機值，如此一來下次在進行訓練時，隨機出來的tenspr就會是相同的!!
lr = 0.001

cnn = CNN()
use_cuda = True
if use_cuda and torch.cuda.is_available():
    cnn.cuda()
#load 先前的model
model_save_name = 'params_cnn_2019_9_13.pkl'
path = F"/content/gdrive/My Drive/{model_save_name}" 
cnn.load_state_dict(torch.load(path))

loss_func = nn.CrossEntropyLoss() # 已經內涵softmax所以不需要另外加
optimizer = torch.optim.Adam(cnn.parameters(), lr = lr)   # optimize all cnn parameters(控制模型學習率的變化)


#%%
acc = []
losses = []
acces = []
eval_losses = []
eval_acces = []
ys_cnn = []
yl_cnn = []
pre = []
lab = []
#def adjust_learning_rate(optimizer, lr):
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr
#    adjust_learning_rate(optimizer, epoch)
for e in range(1):
    train_loss = 0
    train_acc = 0
    cnn.train()
    for im, label in train_loader:
        if e > 30:
          lr = 0.00001
          adjust_learning_rate(optimizer, lr)
        print(optimizer)
        if use_cuda and torch.cuda.is_available():
          im = Variable(im).cuda()
          label = Variable(label).cuda()
        # 前向傳播
        out = cnn(im)
        loss = loss_func(out, label)
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 紀錄誤差
        train_loss += loss.item()
        # 計算準確率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
#    # 檢驗結果
    eval_loss = 0
    eval_acc = 0
    cnn.eval() # 改為預測模式
    for im, label in test_loader:
        if use_cuda and torch.cuda.is_available():
          im = Variable(im).cuda()
          label = Variable(label).cuda()
        out = cnn(im)
        loss = loss_func(out, label)
        # 紀錄誤差
        eval_loss += loss.item()
        # 計算準確率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        ys_cnn.append(out.cpu().data.numpy())
        yl_cnn.append(label.cpu().numpy())
        pre.append(pred)
        lab.append(label)
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_loader), train_acc / len(train_loader), 
                     eval_loss / len(test_loader), eval_acc / len(test_loader)))
# 儲存訓練好的model
    model_save_name = 'params_cnn_2019_9_13.pkl'
    path = F"/content/gdrive/My Drive/{model_save_name}" 
    torch.save(cnn.state_dict(), path)
