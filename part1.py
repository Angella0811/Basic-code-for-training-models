
import matplotlib_inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
from time import time
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples,num_inputs,dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)  # 标签加入随机噪声
print(features[0], labels[0])

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j)

batch_size = 3

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data


lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss
for epoch in range(num_epochs): # 共需要num_epochs个迭代周期
   for X ,y in data_iter(batch_size,features,labels):
       l = loss(net(X, w, b), y).sum()
       l.backward()
       sgd([w,b],lr,batch_size)
       w.grad.data.zero_()
       b.grad.data.zero_()
   train_l = loss(net(features,w,b),labels)
   print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
