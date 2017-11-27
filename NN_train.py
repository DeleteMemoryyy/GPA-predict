# -*- coding: UTF-8 -*-
import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.preprocessing as prep
import sklearn.svm as svm
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
import seaborn as sns
from collections import OrderedDict
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as Functional
import torch.nn.init as Init

net_dir = 'net'
temp_save_name = 'temp_save_param.pkl'
C_Batch = 50
C_LR = 0.001
C_Momentum = 0.1
early_stop = True
early_stop_flag = False
stop_mse = 0.140
stop_process = 0.2
tolerancing = 0.05
C_Epoch = 10000
save_gap = 10
print_gap = 10


all_data = pd.read_csv('data/ALLDATA.csv')

drop_columns = ['grade', 'admit_grade', 'high_school', 'high_rank',
                'rank_var', 'progress', 'patent', 'social', 'prize', 'competition']

one_hot_columns = ['province', 'gender', 'birth_year', 'nation', 'politics', 'color_blind',
                   'stu_type', 'lan_type', 'sub_type', 'test_year', 'department', 'reward_type']

numerical_columns = ['left_sight', 'right_sight', 'admit_rank', 'school_num', 'center_grade'
                     'school_admit_rank', 'reward_score', 'height', 'weight']

standardization_columns = ['left_sight', 'right_sight', 'admit_rank', 'school_num',
                           'school_admit_rank', 'reward_score', 'height', 'weight']

other_columns = ['student_ID', 'GPA', 'test_tag', 'test_ID']

all_data['rank_var'] = all_data['rank_var'].fillna(all_data['rank_var'].mean())
all_data['high_rank'] = all_data['high_rank'].fillna(
    all_data['high_rank'].mean())
all_data['progress'] = all_data['progress'].fillna(all_data['progress'].mean())
all_data['admit_grade'] = all_data['admit_grade'].fillna(
    all_data['admit_grade'].mean())
all_data['center_grade'] = all_data['center_grade'].fillna(0)
all_data['reward_score'] = all_data['reward_score'].fillna(0)
all_data['prize'] = all_data['prize'].fillna(0)
all_data['competition'] = all_data['competition'].fillna(0)
all_data['patent'] = all_data['patent'].fillna(0)
all_data['social'] = all_data['social'].fillna(0)

proc_data = all_data

# one-hot processing
proc_data[one_hot_columns] = proc_data[one_hot_columns].fillna('Empty')
proc_data = pd.merge(proc_data, pd.get_dummies(
    proc_data, columns=one_hot_columns))
proc_data = proc_data.drop(one_hot_columns, axis=1)

# process sight


def precess_sight(x):
    if (x == np.nan):
        return x
    if(x >= 3.0):
        return x
    elif (x >= 2.0 and x < 3.0):
        return 5.3
    elif (x >= 1.5 and x < 2.0):
        return 5.2
    elif (x >= 1.2 and x < 1.5):
        return 5.1
    elif (x >= 1.0 and x < 1.2):
        return 5.0
    elif (x >= 1.0 and x < 1.2):
        return 5.0
    elif (x >= 0.8 and x < 1.0):
        return 4.9
    elif (x >= 0.6 and x < 0.8):
        return 4.8
    elif (x >= 0.5 and x < 0.6):
        return 4.7
    elif (x >= 0.4 and x < 0.5):
        return 4.6
    elif (x >= 0.3 and x < 0.4):
        return 4.5
    elif (x >= 0.25 and x < 0.3):
        return 4.4
    elif (x >= 0.2 and x < 0.25):
        return 4.3
    elif (x >= 0.15 and x < 0.2):
        return 4.2
    elif (x >= 0.12 and x < 0.15):
        return 4.1
    elif (x >= 0.1 and x < 0.12):
        return 4.0
    elif (x >= 0.08 and x < 0.1):
        return 3.9
    elif (x >= 0.08 and x < 0.1):
        return 3.9
    elif (x >= 0.06 and x < 0.08):
        return 3.8
    elif (x >= 0.05 and x < 0.06):
        return 3.7
    elif (x >= 0.04 and x < 0.05):
        return 3.6
    elif (x >= 0.03 and x < 0.04):
        return 3.5
    else:
        return np.nan


proc_data['left_sight'] = proc_data['left_sight'].apply(precess_sight)
proc_data['right_sight'] = proc_data['right_sight'].apply(precess_sight)
proc_data['left_sight'] = proc_data['left_sight'].fillna(
    proc_data['left_sight'].mean())
proc_data['right_sight'] = proc_data['right_sight'].fillna(
    proc_data['right_sight'].mean())

# drop features
proc_data = proc_data.drop(drop_columns, axis=1)
proc_data = proc_data.drop(other_columns, axis=1)

#%% standardization
ss_x = prep.StandardScaler()
proc_data[standardization_columns] = ss_x.fit_transform(
    proc_data[standardization_columns].values)

#%% spilt training data
x_all_train = proc_data[all_data['test_tag'] != 'test']
y_all_train = all_data['GPA'][all_data['test_tag'] != 'test']
x_test = proc_data[all_data['test_tag'] == 'test']
x_train, x_valid, y_train, y_valid = train_test_split(x_all_train.values, y_all_train.values,
                                                      random_state=33)

d_feature = x_all_train.shape[1]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(OrderedDict([
            ('hidden1', nn.Linear(d_feature, 1024)),
            # ('bn1',nn.BatchNorm1d(1024)),
            ('drop1',nn.Dropout(0.5)),
            ('relu1', nn.ReLU()),
            ('hidden2', nn.Linear(1024, 512)),
            # ('bn2',nn.BatchNorm1d(5112)),
            ('drop2',nn.Dropout(0.5)),
            ('relu2', nn.ReLU())
        ]))
        Init.xavier_uniform(self.hidden.hidden1.weight,gain=np.sqrt(2.0))
        Init.xavier_uniform(self.hidden.hidden2.weight,gain=np.sqrt(2.0))
        self.out = nn.Linear(512, 1)

    def forward(self, x):
        hidden = self.hidden(x)
        out = self.out(hidden)
        return out

net = Net()

if(torch.cuda.is_available()):
    net = net.cuda()
    print('Using gpu')
else:
    print('Using cpu')
print(net)

torch_x_train = torch.from_numpy(x_train).float()
torch_y_train = torch.from_numpy(y_train).float()
torch_data = Data.TensorDataset(data_tensor=torch_x_train,target_tensor=torch_y_train)
train_data = Data.DataLoader(dataset=torch_data,batch_size=C_Batch,shuffle=True)
torch_x_valid = torch.from_numpy(x_valid).float()
if(torch.cuda.is_available()):
    torch_x_valid = torch_x_valid.cuda()
v_x_valid = Variable(torch_x_valid)

optimizer = torch.optim.SGD(net.parameters(),lr=C_LR,momentum=C_Momentum)
loss_func = nn.MSELoss()
loss = 1.0

time_s = time.time()
time_u = time_s
epoch = 0
recent_loss = 100
train_err_list = []
valid_err_list = []
learning_time = []
for epoch in range(C_Epoch):
    for step,(x,y) in enumerate(train_data):
        if (torch.cuda.is_available()):
            x = x.cuda()
            y = y.cuda()

        v_x = Variable(x.view(-1,d_feature))
        v_y = Variable(y.view(-1,1)).float()
        optimizer.zero_grad()

        y_train_out = net(v_x)
        loss = loss_func(y_train_out,v_y)
        reg_loss = loss + ((net.hidden.hidden1.weight ** 2).sum() + (net.hidden.hidden2.weight ** 2).sum() + (net.out.weight ** 2).sum()) * 0.05 / x.shape[0]
        reg_loss.backward()
        optimizer.step()

    time_c = time.time()
    if ((epoch % print_gap) == 0):
        print('Process = {0:.3f}%'.format((float(epoch+print_gap)/float(C_Epoch))*100.0))
        print('Training time = {0:.3f}s, this round = {1:.3f}s, remaining time = {2:.3f}s'.format(time_c-time_s,time_c-time_u,(time_c-time_u)*(C_Epoch-epoch-1)/float(print_gap)))
        print('Training mse = {0:.4f}'.format(loss.cpu().data.numpy()[0]))
        time_u = time_c
        if (not((epoch % save_gap) == 0)):
            print('')

    if ((epoch % save_gap) == 0):
        y_valid_out = net(v_x_valid).cpu().data.numpy()
        this_loss = metrics.mean_squared_error(y_valid,y_valid_out)
        train_err_list.append(loss.cpu().data.numpy()[0])
        valid_err_list.append(this_loss)
        learning_time.append(epoch + 1)
        print('Validating mse = {0:.4f}'.format(this_loss))
        if(early_stop and epoch > C_Epoch * stop_process and recent_loss <= stop_mse
           and this_loss > (recent_loss * (1 + tolerancing))):
            early_stop_flag = True
            epoch -= save_gap
            print('Early stop at epoch: {0:d}/{1:d}\n'.format(epoch,C_Epoch))
            break
        recent_loss = this_loss
        torch.save(net.cpu().state_dict(),'{0:s}/{1:s}'.format(net_dir,temp_save_name))
        if(torch.cuda.is_available()):
            net = net.cuda()
        print('Temporary save to {0:s}/{1:s} at {2:d}/{3:d}'.format(net_dir,temp_save_name,epoch + 1,C_Epoch))
        print('')

print('Training finish!')
print('Finishing loss: {0:f}'.format(loss.cpu().data.numpy()[0]))

if (not os.path.exists(net_dir)):
    os.mkdir(net_dir)
existed_nets = os.listdir(net_dir)
os.chdir(net_dir)
counter = 0
for e_net in existed_nets:
    if (os.path.isfile(e_net) and e_net.endswith('_param.pkl')):
        counter += 1
os.chdir("..")
save_name = '{0:d}_{1:.4f}_{2:d}_{3:d}_net_param.pkl'.format(counter,C_LR,epoch + 1,C_Batch)
if(early_stop_flag):
    os.rename('{0:s}/{1:s}'.format(net_dir,temp_save_name),'{0:s}/{1:s}'.format(net_dir,save_name))
else:
    torch.save(net.cpu().state_dict(),'{0:s}/{1:s}'.format(net_dir,save_name))
print('Save to: {0:s}/{1:s}'.format(net_dir,save_name))

# plot loss history
plt.plot(learning_time, train_err_list, 'r-')
plt.plot(learning_time, valid_err_list, 'b--')
plt.show()