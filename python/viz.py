
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 500)


# In[2]:


d = pd.read_csv("trainv2_clean.csv")


# In[3]:


d['bought'] = d.transactions.apply(lambda x: 0 if x==0 else 1)


# In[4]:


bs = list(d.browser.value_counts()[:25].index)


# In[31]:


fig = plt.figure(1, figsize=(16,9))
ax = plt.subplot(111)
sns.countplot(d[d['browser'].isin(bs)]['browser'], hue=d[d['browser'].isin(bs)]['bought'], ax=ax)
plt.xticks(rotation=60)
plt.title('Buyers by top 10 Browsers')
plt.show()


# In[62]:


fig = plt.figure(1, figsize=(16,9))
ax = plt.subplot(111)
sns.countplot(d[d['browser'].isin(bs)]['isMobile'], hue=d[d['browser'].isin(bs)]['bought'], ax=ax)
plt.xticks(rotation=60)
plt.title('Buyers by Mobile')
plt.show()


# In[34]:


fig = plt.figure(1, figsize=(16,9))
ax = plt.subplot(111)
sns.countplot(d['operatingSystem'], hue=d['bought'], ax=ax)
plt.xticks(rotation=60)
plt.title('Buyers by Operating System')
plt.show()


# In[36]:


fig = plt.figure(1, figsize=(16,9))
ax = plt.subplot(111)
sns.countplot(d['continent'], hue=d['bought'], ax=ax)
plt.xticks(rotation=60)
plt.title('Buyers by Continent')
plt.show()


# In[66]:


fig = plt.figure(1, figsize=(16,9))
ax = plt.subplot(111)
sns.boxplot(d['timeOnSite'], hue=d['bought'], ax=ax)
plt.xticks(rotation=60)
plt.title('Boxplot of TimeOnSite')
plt.show()

