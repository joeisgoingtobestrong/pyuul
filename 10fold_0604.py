#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch, os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pyuul import utils, VolumeMaker
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib.pyplot import MultipleLocator
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import ndcg_score


# In[21]:


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        data = []
        label = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], float(words[1])))
            data.append(words[0])
            label.append(float(words[1]))
    
        self.data = data
        self.label = label
       

    def __getitem__(self, index): 
        fn = self.data[index]
        label = self.label[index]
        label = np.array(label)
        label = label.reshape(-1, 1)
        label = torch.tensor(label, dtype=torch.float32)
        
        return fn, label

    def __len__(self):
        return len(self.data)


# In[22]:


kf = KFold(n_splits=3,shuffle=False)


# In[23]:


#改路径
#train_data = MyDataset(txt_path='/lustre/home/acct-clschf/clschf/yxjiang/trajin/pyuul/57/activ127.txt')
train_data = MyDataset(txt_path='/Volumes/TOSHIBA/pyuul/523/activ12.txt')


# In[24]:


###读测试集###
#改路径
#test_data = MyDataset(txt_path='/lustre/home/acct-clschf/clschf/yxjiang/trajin/pyuul/57/activ127.txt')
test_data = MyDataset(txt_path='/Volumes/TOSHIBA/pyuul/523/activ_test_local.txt')


# In[25]:


for i, data in enumerate(test_data):
            test_fn, test_labels = data
            print(test_fn)


# In[26]:


train_dataset=[]
valid_dataset=[]
test_dataset=[]


# In[27]:


###分割训练集和验证集###
#改
for train_index , valid_index in kf.split(train_data):  # 调用split方法切分数据
    print('train_index:%s , test_index: %s ' %(train_index,valid_index))
    ten_fold_index=0
    train_dataset.append(train_index)
    valid_dataset.append(valid_index)
    ten_fold_index=ten_fold_index+1


# In[ ]:





# In[28]:


def num_one(source_array):
    count=0
    for x in source_array:
        count+=1
    return count


# In[29]:


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
            # torch.nn.Linear(1000,100),
            # torch.nn.Linear(100,final_outchannels),
            torch.nn.Linear(final_outchannels, hiddenLayer),
            torch.nn.LayerNorm(hiddenLayer),
            torch.nn.Dropout(dropoutProb),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenLayer, hiddenLayer),
            torch.nn.LayerNorm(hiddenLayer),
            torch.nn.Dropout(dropoutProb),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenLayer, numberofFeatures),
            # torch.nn.ReLU()
        )

        self.linear = torch.nn.Sequential(
            # torch.nn.Linear(1000,100),
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
            torch.nn.Conv3d(outchannel, outchannel, kernel_size=(6, 5, 6), stride=2, padding=2),
            torch.nn.Conv3d(outchannel, outchannel, kernel_size=(11, 1, 12), stride=2, padding=4),
            torch.nn.Conv3d(outchannel, final_outchannels, kernel_size=(9, 9, 9), stride=3, padding=2)

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
            torch.nn.Conv3d(outchannel, outchannel, kernel_size=(6, 5, 6), stride=2, padding=2),
            torch.nn.Conv3d(outchannel, outchannel, kernel_size=(11, 1, 12), stride=2, padding=4),
            torch.nn.Conv3d(outchannel, final_outchannels, kernel_size=(9, 9, 9), stride=3, padding=2)
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
        pred = self.predict(orientedx)
        channels = pred.shape[1]
        pred = pred.view(batch_size, channels, -1)
        pred = pred.sum(-1)
        
        return self.final_net(pred)


# In[ ]:





# In[30]:


Model = RegressionArchitecture(6)
#device = torch.device("cuda:0")
#Model = Model.to(device)


# In[31]:


optimizer = torch.optim.Adam(Model.parameters())
lossFunction = torch.nn.MSELoss()
#device = torch.device("cuda:0")
#lossFunction = lossFunction.to(device)


# In[95]:


from datetime import datetime

time = datetime.now()
name_date = f'{str(time.month).zfill(2)}{str(time.day).zfill(2)}'


# In[45]:


###用于记录loss的均值、次数和总和###
#class Counter:
#    def __init__(self):
#        self.count, self.sum, self.avg = 0, 0, 0
#        return

#    def update(self, value, num_updata=1):
#        self.count += num_updata
#        self.sum += value * num_updata
#        self.avg = self.sum / self.count
#        return

#    def clear(self):
#        self.count, self.sum, self.avg = 0, 0, 0
#        return
    
    
    


# In[46]:


###用于保存loss###
#class Loss_Saver:
#    def __init__(self, moving=False):
#        self.loss_list, self.last_loss = [], 0.0
#        self.moving = moving  # 是否进行滑动平均操作
	
#    def updata(self, value):
		# 只有进行滑动平均时，才会用到 self.last_loss 
		
#        if not self.moving:
#            self.loss_list += [value]
#        elif not self.loss_list:
#            self.loss_list += [value]
#            self.last_loss = value
#        else:
#            update_val = self.last_loss * 0.9 + value * 0.1
#            self.loss_list += [[update_val]]
#            self.last_loss = update_val
#        return


#    def loss_drawing(self, root_file, encoding='gbk'):
    
    	# 这个用于在指定位置保存 loss 指标
#        loss_array = np.array(self.loss_list)
       
#        colname = ['loss']
#        listPF = pd.DataFrame(columns=colname, data=loss_array)
#        listPF.to_csv(f'{root_file}loss.csv', encoding=encoding)
    


# In[121]:


def loss_drawing(epoch_list,loss_list,loss_avg_list,root_file, encoding='gbk'):
    
    	# 这个用于在指定位置保存 loss 指标
    loss_array = np.array(loss_list)
        
    dict1 = {"epoch":epoch_list,
         "loss_sum":loss_array,
            "loss_avg":loss_avg_list}
    df = pd.DataFrame(dict1)
    df.to_csv(f'{root_file}.csv', encoding=encoding)
    #return df3
    
def pred_drawing(sample,output_list,true,loss,root_file,encoding='gbk'):
    output_array=np.array(output_list)
    dict2={"sample":sample,
          "pred":output_array,
          "true":true,
          "loss":loss}
    df = pd.DataFrame(dict2)
    df.to_csv(f'{root_file}.csv',encoding=encoding)
    #return df


# In[53]:


train_dataset_k


# In[126]:


###模型训练###
for train_dataset_i  in range(3):#modify
    ###列表初始化###
    epoch_list=[]
    train_loss_list=[]
    train_loss_avg=[]
    epoch_list=[]
    
    valid_sample=[]
    valid_output=[]
    valid_labels=[]
    valid_loss_list=[]
    
    test_sample=[]
    test_output=[]
    
    test_labels=[]

    
    test_loss_list=[]
    
    Model.train(mode=True)
    for epoch in range(2):#epoch modify
        train_dataset_k=0
        train_loss=0
        for i,data in enumerate(train_data):
            if (train_dataset_k<num_one(train_dataset[0]) and i == train_dataset[train_dataset_i][train_dataset_k] ) :
                train_dataset_k+=1
            else:
                continue
            
            fn, labels = data
            #labels=labels.to(device)
            #device = torch.device("cuda:0")
            coords, atname = utils.parsePDB(str(fn)) #一次读取所有的结构
            atom_channel = utils.atomlistToChannels(atname)#原子通道
            radius = utils.atomlistToRadius(atname)#原子所占体素空间
            VoxelsObject = VolumeMaker.Voxels(sparse=False)
            #VoxelsObject = VoxelsObject.to(device)
            voxellizedVolume = VoxelsObject(coords, radius, atom_channel,resolution=0.5) 
            #voxellizedVolume=voxellizedVolume.to(device)
            
            yp = Model(voxellizedVolume)
            loss = lossFunction(yp, labels)
            #loss=loss.to(device)

            train_loss+=float(loss.item())
            
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
            #print("epoch",epoch, "loss ", float(loss.data.cpu()))
        
            #print(epoch_loss%2)
        
        epoch_list.append(int(epoch))
        epoch+=1
        train_loss_list.append(train_loss)
        train_loss_avg.append(train_loss/int(train_dataset_k))
######留一验证######  
        Model.eval()
        with torch.no_grad():
            valid_dataset_k=0
            for i, data in enumerate(train_data):        
                if (valid_dataset_k<num_one(valid_dataset[0]) and i == valid_dataset[train_dataset_i][valid_dataset_k]):
                    valid_dataset_k+=1 
                else:
                    continue

                fn, valid_label = data
                #valid_labels=labels.to(device)

                #device = torch.device("cuda:0")
                valid_coords, valid_atname = utils.parsePDB(str(fn)) #一次读取所有的结构
                valid_atom_channel = utils.atomlistToChannels(valid_atname)#原子通道
                valid_radius = utils.atomlistToRadius(valid_atname)#原子所占体素空间
                valid_VoxelsObject = VolumeMaker.Voxels(sparse=False)
                #valid_VoxelsObject = valid_VoxelsObject.to(device)
                valid_voxellizedVolume = valid_VoxelsObject(valid_coords, valid_radius, valid_atom_channel,resolution=0.5)#构建体素表示
                #valid_voxellizedVolume=valid_voxellizedVolume.to(device)

    ###计算测试集loss和output#####       
                valid_loss=0
                output=Model(valid_voxellizedVolume)
                valid_loss=lossFunction(valid_label,output)
                #valid_loss=test_loss.to(device)

                #print("第", train_dataset_i, "折的第",valid_dataset_k,"个样本的loss", float(valid_loss.data.cpu()))
                #print("样本：",str(fn),"预测结果：", output,"真实值：",labels)
                #output_list.append("第", train_dataset_i, "折","样本：",str(fn),"预测结果：", output,"真实值：",labels)


                valid_sample.append(str(fn))
                valid_output.append(float(output))
                valid_labels.append(float(valid_label))
                valid_loss_list.append(float(valid_loss.item()))
            
    ###测试集###
    

    #torch.save(Model.state_dict(), "/Volumes/TOSHIBA/pyuul/523/output/{}fold_model_parameter.pkl".format((train_dataset_i)))
    torch.save(Model, "/Volumes/TOSHIBA/pyuul/523/output/{}_{}fold_model.pkl".format(name_date,train_dataset_i)) 
    #print('model.state_dict:',Model.state_dict())
    with torch.no_grad():
        for i, data in enumerate(test_data):
            test_fn, test_label = data
            print("test_label_a:",test_label)

            #test_labels=labels.to(device)
            #device = torch.device("cuda:0")
            test_coords, test_atname = utils.parsePDB(str(test_fn)) #一次读取所有的结构
            test_atom_channel = utils.atomlistToChannels(test_atname)#原子通道
            test_radius = utils.atomlistToRadius(test_atname)#原子所占体素空间
            test_VoxelsObject = VolumeMaker.Voxels(sparse=False)
            #test_VoxelsObject = test_VoxelsObject.to(device)
            test_voxellizedVolume = test_VoxelsObject(test_coords, test_radius, test_atom_channel,resolution=0.5)#构建体素表示
            #test_voxellizedVolume=test_voxellizedVolume.to(device)
            

            test_loss=0 
            pred=Model(test_voxellizedVolume)
            test_loss=lossFunction(test_label,pred)
            
            #test_loss=test_loss.to(device)

            #print("样本：",str(test_fn),"预测结果：", output,"真实值：",test_labels,"loss", float(test_loss.data.cpu()))
            #output_list.append("第", train_dataset_i, "折","样本：",str(fn),"预测结果：", output,"真实值：",labels)

            test_sample.append(str(test_fn))
            test_output.append(float(pred))
            test_labels.append(float(test_label))
            test_loss_list.append(float(test_loss.item()))
   
                              

    root_exp_file = "/Volumes/TOSHIBA/pyuul/523/output/"
    
    
    #torch.save(Model.state_dict(), "/Volumes/TOSHIBA/pyuul/523/output/{}fold_model_parameter.pkl".format((train_dataset_i)))
    
    #print('save{}fold_model'.format(train_dataset_i))
    #print('model.state_dict:',Model.state_dict())
    
    train_name_exp=str('第{}折train_loss'.format(train_dataset_i))
    train_loss=loss_drawing(epoch_list=epoch_list,loss_list=train_loss_list,loss_avg_list=train_loss_avg,root_file=f'{root_exp_file}/{name_date}_{train_name_exp}')
    print(("train_save_finished")
    
    valid_name_exp=str('第{}折valid_output'.format(train_dataset_i))
    valid_file=pred_drawing(sample=valid_sample,output_list=valid_output,true=valid_labels,loss=valid_loss_list,root_file=f'{root_exp_file}/{name_date}_{valid_name_exp}')
    print(("valid_save_finished")
                              
    test_name_exp=str('第{}折test_output'.format(train_dataset_i))
    test_file=pred_drawing(sample=test_sample,output_list=test_output,true=test_labels,loss=test_loss_list,root_file=f'{root_exp_file}/{name_date}_{test_name_exp}')
    print("test_save_finished")
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[103]:



new_model1 = RegressionArchitecture(6)                                                  # 调用模型Model
new_model1.load_state_dict(torch.load("/Volumes/TOSHIBA/pyuul/523/output/0fold_model_parameter.pkl"))    # 加载模型参数     
print(test_fn)
print(test_label)
    #Model.eval()
    pred=new_model1(test_voxellizedVolume)
    print(pred)


# In[104]:



new_model2 = RegressionArchitecture(6)                                                  # 调用模型Model
new_model2.load_state_dict(torch.load("/Volumes/TOSHIBA/pyuul/523/output/1fold_model_parameter.pkl"))    # 加载模型参数     
print(test_fn)
print(test_label)

pred=new_model2(test_voxellizedVolume)
print(pred)


# In[105]:



new_model3 = RegressionArchitecture(6)                                                  # 调用模型Model
new_model3.load_state_dict(torch.load("/Volumes/TOSHIBA/pyuul/523/output/2fold_model_parameter.pkl"))    # 加载模型参数     
print(test_fn)
print(test_label)
#Model.eval()
pred=new_model3(test_voxellizedVolume)
print(pred)


# In[77]:


test_fn


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




