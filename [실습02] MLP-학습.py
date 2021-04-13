#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np


# In[2]:


mnist_train=dset.MNIST("", train=True,transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test=dset.MNIST("", train=False,transform=transforms.ToTensor(), target_transform=None, download=True)


# In[3]:


print "mnist_train 길이: ", len(mnist_train)
print "mnist_test 길이: ", len(mnist_test)

image, label = mnist_test.__getitem__(0)
print "imange data 형태:", image.size()
print "label: ", label

img = image.numpy()
plt.title("label: %d" %label )
plt.imshow(img[0], cmap='gray')
plt.show()


# In[4]:


batch_size = 1024
learning_rate = 0.01
num_epoch = 400


# In[5]:


train_loader = torch.utils.data.DataLoader(mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=2,
                                          drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                          shuffle=False, num_workers=2,
                                          drop_last=True)


# In[6]:


n=3
for i, [imgs, labels] in enumerate(test_loader):
    if i>5:
        break
        
    print "[%d]" %i
    print "한 번에 로드되는 데이터 크기:", len(imgs)
    
    for j in range(3):
        img = imgs[j].numpy()
        img = img.reshape((img.shape[1], img.shape[2]))
        
        plt.subplot(1, n, j+1)
        plt.imshow(img, cmap='gray')
        plt.title("label: %d" %labels[j])
    plt.show()


# In[7]:


model = nn.Sequential(
    nn.Linear(28*28,256),
    nn.Sigmoid(),
    nn.Linear(256,128),
    nn.Linear(128,10),
)


# In[9]:


def ComputeAccr(dloader, imodel):
    correct = 0
    total = 0
    
    for j, [imgs, labels] in enumerate(dloader):
        img = imgs
        label = Variable(labels)
        
        img = img.reshape((img.shape[0], img.shape[2], img.shape[3]))
        
        img = img.reshape((img.shape[0], img.shape[1]*img.shape[2]))
        img = Variable(img, requires_grad=False)
        
        output = imodel(img)
        _, output_index = torch.max(output, 1)
            
        total += label.size(0)
        correct += (output_index == label).sum().float()
    print("Accuracy of Test Data: {}".format(100*correct/total))


# In[10]:


ComputeAccr(test_loader, model)


# In[11]:


loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# In[13]:


for i in range(num_epoch):
    for j, [imgs, labels] in enumerate(train_loader):
        img = imgs
        label = Variable(labels)
        
        img = img.reshape((img.shape[0], img.shape[2], img.shape[3]))
        
        img = img.reshape((img.shape[0], img.shape[1]*img.shape[2]))
        img = Variable(img, requires_grad=True)
        
        optimizer.zero_grad()
        output = model(img)
        loss = loss_func(output, label)
        
        loss.backward()
        optimizer.step()
        
    if i%50==0:
        print("%d.." %i)
        ComputeAccr(test_loader, model)
        print loss


# In[14]:


ComputeAccr(test_loader, model)


# In[16]:


netname = './nets/mlp_weight.pkl'
torch.save(model, netname, )

