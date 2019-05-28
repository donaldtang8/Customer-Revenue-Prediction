
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json
from ast import literal_eval

pd.set_option('display.max_columns', 500)


# In[2]:


d = pd.read_csv("/wdblue/gstore_spending/all/train_v2.csv")


# In[36]:


# for i in range(600):
#     x = json.loads(d.hits[i])
#     print(x.keys())
#     col = 'networkLocation'
#     if(col in x):
#         print(x[col])
#     else:
#         print("err")


# In[41]:


literal_eval(d.hits[1])


# In[42]:


for i in range(10):
    for hit in literal_eval(d.hits[i]):
        print(hit.keys())


# In[43]:


# Fields in json objects to keep, others are "not included in the demo dataset"
device_list = ['browser', 'operatingSystem', 'isMobile', 'deviceCategory']
geo_nets = ['continent', 'subContinent', 'country', 'region', 'city', 'networkDomain']
totals_fields = ['visits', 'hits_total', 'pageviews', 'bounces', 'newVisits', 'sessionQualityDim', 
                 'timeOnSite', 'transactions', 'transactionRevenue', 'totalTransactionRevenue']
traffic_fields = ['campaign', 'source', 'medium', 'keyword', 'referralPath', 'isTrueDirect']


# In[44]:


d.drop('socialEngagementType', axis=1, inplace=True)
d.drop('date', axis=1, inplace=True)


# In[45]:


for device in device_list:
    d[device] = d.device.apply(lambda x: json.loads(x)[device])
d.drop('device', axis=1, inplace=True)


# In[46]:


for geo in geo_nets:
    d[geo] = d.geoNetwork.apply(lambda x: json.loads(x)[geo])
d.drop('geoNetwork', axis=1, inplace=True)


# In[47]:


for tot in totals_fields:
    d[tot] = d.totals.apply(lambda x: int(json.loads(x)[tot]) if tot in json.loads(x) else 0)
d.drop('totals', axis=1, inplace=True)


# In[48]:


d['customDim'] = d.customDimensions.apply(lambda x: literal_eval(x[1:len(x)-1])['value'] if x != '[]' else 'NA')
d.drop('customDimensions', axis=1, inplace=True)


# In[49]:


for field in traffic_fields:
    if(field=='isTrueDirect'):
        d[field] = d.trafficSource.apply(lambda x: json.loads(x)[field] if field in json.loads(x) else False)
    else:
        d[field] = d.trafficSource.apply(lambda x: json.loads(x)[field] if field in json.loads(x) else '[NOT GIVEN]')
d.drop('trafficSource', axis=1, inplace=True)


# In[50]:


d.head()


# In[56]:


d.drop(['hits', 'hits_total', 'visits'], axis=1, inplace=True)
d.to_csv('trainv2_clean.csv', index=False)
print(d.head())

