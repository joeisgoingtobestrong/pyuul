#!/usr/bin/env python
# coding: utf-8

# In[157]:


import torch, os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pyuul import utils, VolumeMaker
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib.pyplot import MultipleLocator


# In[144]:


#os.chdir('/Volumes/TOSHIBA/pyuul')


# In[ ]:





# In[145]:


from PIL import Image
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        data=[]
        label=[]
        for line in fh:
            line = line.rstrip()
            words = line.split()
            #imgs.append((words[0], int(words[1])))
            imgs.append((words[0], float(words[1])))
            data.append(words[0])
            label.append(float(words[1]))
            #imgs.append((words[0], words[1]))
        #print(imgs)
        #self.imgs = imgs 
        self.data=data
        self.label=label
            #self.transform = transform
            #self.target_transform = target_transform
            #self.voxellizedVolume=voxellizedVolume
        
        
    def __getitem__(self, index):
        #fn, label = self.imgs[index]
        fn=self.data[index]
        label=self.label[index]
        #print('fnindataset',fn)
        #img = Image.open(fn).convert('RGB') 
        #if self.transform is not None:
            #img = self.transform(img) 
        label = np.array(label)
        label = label.reshape(-1,1)
        label = torch.tensor(label,dtype=torch.float32)
    # device='cpu'
        #coords, atname = utils.parsePDB(fn) #一次读取所有的结构
        #atom_channel = utils.atomlistToChannels(atname)#原子通道
        #radius = utils.atomlistToRadius(atname)#原子所占体素空间
        #VoxelsObject = VolumeMaker.Voxels(device=device, sparse=False)
        #voxellizedVolume = VoxelsObject(coords, radius, atom_channel,resolution=0.5) 
        #print('voxellizedVolume',voxellizedVolume.shape)
        return fn, label
    def __len__(self):
        return len(self.data)


# In[ ]:


test_label=pd.read_csv('/lustre/home/acct-clschf/clschf/yxjiang/trajin/pyuul/therm.csv',header=None)
#modify,the values of testset

# In[147]:


test_labels=np.array(test_label.iloc[:,1])


# In[148]:


test_labels = test_labels.reshape(-1,1)
test_labels = torch.tensor(test_labels,dtype=torch.float32)
print(test_labels)


# In[149]:


#device = "cpu" 
test_coords, test_atname = utils.parsePDB("/lustre/home/acct-clschf/clschf/yxjiang/trajin/pyuul/pyuul2/") #modify:一次读取所有的用于测试的结构

test_atom_channel = utils.atomlistToChannels(test_atname)#原子通道
test_radius = utils.atomlistToRadius(test_atname)#原子所占体素空间
#test_VoxelsObject = VolumeMaker.Voxels(device=device, sparse=False)
test_VoxelsObject = VolumeMaker.Voxels(sparse=False)
test_voxellizedVolume = test_VoxelsObject(test_coords, test_radius, test_atom_channel,resolution=0.5)#构建体素表示
print(test_voxellizedVolume.shape)#判断原子通道实例化模型


# In[ ]:





# In[ ]:


train_data=MyDataset(txt_path='/lustre/home/acct-clschf/clschf/yxjiang/trajin/pyuul/413/therm841.txt')
#modify:a file contains all paths and values of train data

# In[151]:


#train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True )


# In[152]:


class RegressionArchitecture(torch.nn.Module):

    def __init__(self, inchannels, numberofFeatures=1):
        super(RegressionArchitecture, self).__init__()
        dilatation = 4
        stride = 2
        kernel_size = 5
        outchannel = 5
        final_outchannels = 30
        padding = 1
        self.softmax = torch.nn.Softmax(2)
        dropoutProb = 0.1

        hiddenLayer = 40
        self.final_net = torch.nn.Sequential(
            #torch.nn.Linear(1000,100),
            #torch.nn.Linear(100,final_outchannels),
            torch.nn.Linear(final_outchannels, hiddenLayer),
            torch.nn.LayerNorm(hiddenLayer),
            torch.nn.Dropout(dropoutProb),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenLayer, hiddenLayer),
            torch.nn.LayerNorm(hiddenLayer),
            torch.nn.Dropout(dropoutProb),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenLayer, numberofFeatures),
            #torch.nn.ReLU()
        )
        
        self.linear=torch.nn.Sequential(
            #torch.nn.Linear(1000,100),
            torch.nn.Linear(100, 10),
            torch.nn.Linear(10, 1)
        )

        self.predict = torch.nn.Sequential(
            torch.nn.Conv3d(inchannels, outchannel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            torch.nn.Dropout(dropoutProb),
            torch.nn.InstanceNorm3d(outchannel, affine=True),
            torch.nn.Tanh(),
            torch.nn.Conv3d(outchannel, outchannel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            torch.nn.Dropout(dropoutProb),
            torch.nn.InstanceNorm3d(outchannel, affine=True),
            torch.nn.Tanh(),
            torch.nn.Conv3d(outchannel,outchannel,kernel_size=(6,5,6),stride=2,padding=2),
            torch.nn.Conv3d(outchannel,outchannel,kernel_size=(11,1,12),stride=2,padding=4),
            torch.nn.Conv3d(outchannel,final_outchannels,kernel_size=(9,9,9),stride=3,padding=2)
            
        )

        self.attention = torch.nn.Sequential(
            torch.nn.Conv3d(inchannels, outchannel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            torch.nn.Dropout(dropoutProb),
            torch.nn.InstanceNorm3d(outchannel, affine=True),
            torch.nn.Tanh(),
            torch.nn.Conv3d(outchannel, final_outchannels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            torch.nn.Dropout(dropoutProb),
            torch.nn.InstanceNorm3d(final_outchannels, affine=True),
            torch.nn.Tanh(),
            torch.nn.Conv3d(outchannel,outchannel,kernel_size=(6,5,6),stride=2,padding=2),
            torch.nn.Conv3d(outchannel,outchannel,kernel_size=(11,1,12),stride=2,padding=4),
            torch.nn.Conv3d(outchannel,final_outchannels,kernel_size=(9,9,9),stride=3,padding=2)
        )
        kernel1 = 5
        padding = 2
        stride = 1
        kernel2 = 5
        self.channels = 2

        outchannelLocaliz = 5

        self.localizationPred = torch.nn.Sequential(
            torch.nn.Conv3d(inchannels, outchannelLocaliz, kernel_size=kernel1, stride=stride, padding=padding),
            torch.nn.MaxPool3d(2, stride=2),
            torch.nn.InstanceNorm3d(outchannelLocaliz, affine=True),
            torch.nn.Tanh()
        )

        self.localizationAtte = torch.nn.Sequential(
            torch.nn.Conv3d(inchannels, outchannelLocaliz, kernel_size=kernel1, stride=stride, padding=padding),
            torch.nn.MaxPool3d(2, stride=2),
            torch.nn.InstanceNorm3d(outchannelLocaliz, affine=True),
            torch.nn.Tanh()
        )

        self.fc_locSA = torch.nn.Sequential(
            torch.nn.Linear(outchannelLocaliz, 32),
            torch.nn.ReLU(True),
            torch.nn.Linear(32, 3 * 4)
        )
        self.fc_locSA[2].weight.data.zero_()


    def stnSA(self, x):
        xsp = self.localizationPred(x)
        xsp = xsp.view(xsp.shape[0], xsp.shape[1], -1)

        xsa = self.localizationAtte(x)
        xsa = self.softmax(xsa.view(xsa.shape[0], xsa.shape[1], -1))
        fin = (xsp * xsa).sum(-1)

        theta = self.fc_locSA(fin)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        batch_size = x.shape[0]

        vals = x

        orientedx = self.stnSA(vals)
        #print('orientedx.size:',orientedx.size())
       # orientedx = orientedx
        #del x, vals

        pred = self.predict(orientedx)
        channels = pred.shape[1]
        #print('pred1.size:',pred.size())
        pred = pred.view(batch_size, channels, -1)
        #print('pred1.size()',pred.size())
        #pred=pred.view(100,-1)
       # atte = self.attention(orientedx)
        #print('atte1.size:',atte.size())
        #atte = self.softmax(atte.view(batch_size, channels, -1))
        #print('atte2.size:',atte.size())
        #del orientedx
        #whatfin=atten*pred
        #print('whatfin_size:',whatfin.size())
        #print('whatfin-1:',whatfin.size(-1))
        #fin = (atte * pred).sum(-1)
        #print('fin:',fin)
        #print('fin.size:',fin.size())
        #return self.final_net(fin)
        pred=pred.sum(-1)
        #print('pred2.size',pred.size())
        return self.final_net(pred)


# In[153]:


Model = RegressionArchitecture(6)


# In[154]:


optimizer = torch.optim.Adam(Model.parameters())
lossFunction = torch.nn.MSELoss()


# In[ ]:





# In[141]:


#epochs = 5
#for e in range(epochs):

 #   yp = Model(voxellizedVolume)
  #  print(yp.shape)
   # loss = lossFunction(yp, labels)
    #loss.backward()

    #optimizer.step()
    #optimizer.zero_grad()

   # print("epoch",e, "loss ", float(loss.data.cpu()))


# In[161]:


mse_list = []
w_list=[]
for epoch in range(50):
   
    for i, data in enumerate(train_data):
        
        
        
        fn, labels = data
        coords, atname = utils.parsePDB(str(fn)) #一次读取所有的结构
        atom_channel = utils.atomlistToChannels(atname)#原子通道
        radius = utils.atomlistToRadius(atname)#原子所占体素空间
        #VoxelsObject = VolumeMaker.Voxels(device=device, sparse=False)
        VoxelsObject = VolumeMaker.Voxels(sparse=False)
        voxellizedVolume = VoxelsObject(coords, radius, atom_channel,resolution=0.5) 
        
        yp = Model(voxellizedVolume)
        #print(yp.shape)
        loss = lossFunction(yp, labels)
        loss.backward()
        #mse_list.append(loss)
        optimizer.step()
        optimizer.zero_grad()
        xlabel=epoch+1
        #print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
    print("epoch",epoch, "loss ", float(loss.data.cpu()))
    mse_list.append(float(loss))
    w_list.append(xlabel)

#mse_list.append(float(loss))
#w_list.append(epoch)
#mse_list.append(float(loss))
plt.plot(w_list, mse_list,lw=1,label="loss")
plt.ylabel('Loss')
plt.xlabel('epoch')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()
plt.savefig('420therm.jpg')        


# In[162]:


test_loss=0
output=Model(test_voxellizedVolume)
print(output)
test_loss = lossFunction(output, test_labels)
print( "loss ", float(test_loss.data.cpu()))


# In[163]:


list1=output.tolist()
x= range(0,len(list1))
print(x)
print(list1)
list2=test_labels.tolist()
plt.plot(x,list1,'g',label='STD-DTR')
plt.plot(x,list2,'b',label='STD-XGBR')
print(list2)
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.show()
plt.savefig('420therm_test.jpg')
#plt.savefig('therm84_test.jpg')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




