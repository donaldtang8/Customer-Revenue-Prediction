
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', 500)


# In[2]:


df = pd.read_csv('trainv2_clean.csv')


# In[3]:


cont = ['visitNumber', 'visitStartTime', 'sessionQualityDim', 'pageviews', 'transactionRevenue', 'timeOnSite', 
        'transactions', 'transactionRevenue', 'totalTransactionRevenue']
cat = ['newVisits', 'bounces', 'channelGrouping', 'browser', 'operatingSystem', 'isMobile', 
       'deviceCategory', 'continent', 'subContinent', 'country', 'region', 'city', 'networkDomain', 'customDim', 
       'campaign', 'source', 'medium', 'keyword', 'isTrueDirect']


# In[4]:


train_df = pd.DataFrame(np.zeros((len(df), len(cont)+len(cat)+1)), columns=(cont+cat+['fullVisitorId']))


# In[5]:


df['transactionRevenue'] = np.log(df['transactionRevenue'].values)
df = df.replace(np.log(0), 0)
tr = {'mean': 0, 'std': 0}
for col in cont:
    print(col)
    m = df[col].values.mean()
    sd = df[col].values.std()
    if(col == 'transactionRevenue'):
        tr['mean'] = m
        tr['std'] = sd
    train_df[col] = (df[col].values-m) / sd
for col in cat:
    train_df[col] = df[col].values
train_df['fullVisitorId'] = df['fullVisitorId'].values


# In[7]:


train_df.to_csv('trainv2_df.csv', index=False)


# In[54]:


train_df = train_df.sample(frac=1)
test = train_df.sample(36800)
rest = train_df.index.difference(test.index)
rest = train_df.loc[rest, :]


# In[61]:


no_t = train_df.transactions.unique()[0]
bought = rest[rest['transactions']!=no_t]
num_buy = bought.shape[0]
did_not = rest[rest['transactions']==no_t].sample(num_buy*9)
train_10 = pd.concat((bought,did_not)).sample(frac=1)


# In[64]:


train_10['bought'] = np.equal(train_10.transactions.values, no_t).astype(np.uint8)
train_10['bought'] = train_10['bought'].apply(lambda x: 1 if x==0 else 0)


# In[70]:


test['bought'] = np.equal(test.transactions.values, no_t).astype(np.uint8)
test['bought'] = test['bought'].apply(lambda x: 1 if x==0 else 0)
test.drop(['transactionRevenue', 'transactions', 'totalTransactionRevenue', 'networkDomain', 'customDim'], axis=1, inplace=True)
train_10.drop(['transactionRevenue', 'transactions', 'totalTransactionRevenue', 'networkDomain', 'customDim'], axis=1, inplace=True)


# In[68]:


train_10.to_csv("trainv2_10.csv", index=False)
test.to_csv("testv2_10.csv", index=False)


# In[8]:


train, test = pd.read_csv("trainv2_10.csv"), pd.read_csv("testv2_10.csv")


# In[9]:


cont_cols = ['visitNumber', 'visitStartTime', 'pageviews', 'timeOnSite']
cat_cols = ['newVisits', 'bounces', 'channelGrouping', 'browser', 'operatingSystem', 'isMobile', 
            'deviceCategory', 'continent', 'subContinent', 'country', 'campaign', 'source', 'medium', 
            'isTrueDirect', 'region', 'city', 'keyword']


# In[10]:


for cat_col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[cat_col], test[cat_col]]))
    train[cat_col] = le.transform(train[cat_col])
    test[cat_col] = le.transform(test[cat_col])


# In[14]:


train.to_csv("trainv2_10_enc.csv", index=False)
test.to_csv("testv2_10_enc.csv", index=False)

