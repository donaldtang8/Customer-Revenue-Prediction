
# coding: utf-8

# In[1]:


import torch
import torch.nn
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import torch.optim
import numpy.random
import numpy as np
import math
import pandas as pd
import copy
import time
from sklearn.preprocessing import LabelEncoder
from tensorboardX import SummaryWriter

pd.set_option('display.max_columns', 500)


# In[2]:


class Net(torch.nn.Module):

    def __init__(self, cat_cols, cont_cols, embeds):
        super(Net, self).__init__()
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.embed = embeds
        self.embed_szs = [min(50, 1) for x in self.embed]
        
        #Embed the categoricals
        self.embedLayer = torch.nn.ModuleList([torch.nn.Embedding(i, j) for i,j in zip(self.embed, self.embed_szs)])
        
        #normalize the numericals
        self.bn_layer = torch.nn.BatchNorm1d(len(self.cont_cols))
        
        # Linear Layers
        self.fc1 = torch.nn.Linear(np.array(self.embed_szs).sum() + len(cont_cols), 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 2)
    def forward(self, x):
        # Embedding Layer
        cat_encoded = [embedLayer(x[:, i+len(self.cont_cols)].long()) for i, embedLayer in enumerate(self.embedLayer)]
        cat_encoded = torch.cat(cat_encoded, 1)
        cont_normalized = self.bn_layer(x[:, :len(self.cont_cols)])
        x = torch.cat([cat_encoded, cont_normalized], 1)
        
        # Linear Layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), 1)
        return x


# In[3]:


class GDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx, :]
        x = item[['visitNumber', 'visitStartTime', 'sessionQualityDim', 'timeOnSite', 'pageviews', 'newVisits', 
                  'bounces', 'channelGrouping', 'browser', 'operatingSystem', 'isMobile', 'deviceCategory', 
                  'continent', 'subContinent', 'country', 'campaign', 'source', 'medium', 
                  'isTrueDirect']].values.astype(np.float32)
        y = item[['bought']].values.astype(np.float32)
        return {'x': torch.from_numpy(x), 'y': torch.from_numpy(y)}


# In[15]:


fdata = pd.read_csv("trainv2_10_enc.csv")
test_data = pd.read_csv("testv2_10_enc.csv")
# test_data.drop('customDim', axis=1, inplace=True)
# data.drop('customDim', axis=1, inplace=True)

data = pd.concat([fdata[fdata['bought']==1], fdata[fdata['bought']==0].sample(18129)])
data.to_csv("trainv2_50_enc.csv")

# tr = {'mean': 0.22711817076655114, 'std': 2.0037093202285647} 

cont_cols = ['visitNumber', 'visitStartTime', 'sessionQualityDim', 'timeOnSite', 'pageviews']
cat_cols = ['newVisits', 'bounces', 'channelGrouping', 'browser', 'operatingSystem', 'isMobile', 
            'deviceCategory', 'continent', 'subContinent', 'country', 'campaign', 'source', 'medium', 
            'isTrueDirect']

#label encode the categorical variables
# label_encoders = {}
# for cat_col in cat_cols:
#     label_encoders[cat_col] = LabelEncoder()
#     data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])
    
# label_encoders_t = {}
# for cat_col in cat_cols:
#     label_encoders_t[cat_col] = LabelEncoder()
#     test_data[cat_col] = label_encoders_t[cat_col].fit_transform(test_data[cat_col])


#create testing and training set
msk = numpy.random.rand(len(data)) < 0.8
training_data = data[msk]
val_data = data[~msk]

batch_size = 256

train_ds = GDataset(training_data)
val_ds = GDataset(val_data)
test_ds = GDataset(test_data)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=8)
dataloaders = {'train': train_dl, 'val': val_dl}
dataset_sizes = {'train': len(training_data), 'val': len(val_data)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

embeddings = [(pd.concat([fdata[col], test_data[col]])).nunique() for col in cat_cols]
num_epochs=50

outputId = "fullVisitorId"
output = "bought"
net = Net(cat_cols, cont_cols, embeddings)
model = net.to(device)

# model.load_state_dict(torch.load('net0_3.pt'))

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

print(net)


# In[17]:


test_data[test_data['bought']==1].shape, test_data.shape


# In[5]:


best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 50000

writer = SummaryWriter('runs/' + 'net0_9')

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, sample in enumerate(dataloaders[phase]):
            inputs, labels = sample['x'], sample['y']
            inputs = inputs.to(device)
            labels = labels.to(device).long().squeeze()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                vals, preds = outputs.max(1)
                vals = vals.unsqueeze(1)
                num_correct = torch.eq(preds, labels).sum()
                loss = criterion(outputs, labels)
#                 print(preds, labels, num_correct)
                if(i % 100 == 0):
                    print("Loss at step {}: {}".format(i, loss/batch_size))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += num_correct.item()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_accuracy = running_corrects / dataset_sizes[phase]
        
        writer.add_scalar(phase+'loss', epoch_loss, epoch)
        writer.add_scalar(phase+'accuracy', epoch_accuracy, epoch)

        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            print("Accuracy:", epoch_accuracy)
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())


print('Best val Loss: {:4f}'.format(best_loss))

# load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'net0_9.pt')


# In[6]:


net = Net(cat_cols, cont_cols, embeddings)
net.load_state_dict(torch.load('net0_9.pt'))
model = net.to(device)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

model.eval()

tp, tn, fp, fn = 0, 0, 0, 0

for i, sample in enumerate(test_dl):
    inputs, labels = sample['x'], sample['y']
    inputs = inputs.to(device)
    labels = labels.to(device).long().squeeze()
#     labels = labels.long().squeeze()
    outputs = model(inputs)
    vals, preds = outputs.max(1)
    
    for pred, targ in zip(preds, labels):
        pred, targ = pred.item(), targ.item()
        if(pred==1 and targ==1):
            tp += 1
        elif(pred==1 and targ==0):
            fp += 1
        elif(pred==0 and targ==0):
            tn += 1
        else:
            fn += 1
            
print("True positive: {}, False positive: {}, True negative: {}, False negative: {}".format(tp, fp, tn, fn))
print("Accuracy:", (tp+tn)/(tp+tn+fp+fn))


# ### net0_5: Accuracy ~ 94% (1024, 1024, embed 1:1)
# ### net0_6: Accuracy ~ 93.2% (2048, 1024, embed 2:1)
# ### net0_7: Accuracy ~ 93.2%(2048, 1024, embed 1:1)
# ### net0_8: Accuracy ~ 92.3% (256, 64, embed 2:1)
# ### net0_8x: Accuracy ~ 92.1% (512, 256, embed 2:1)
