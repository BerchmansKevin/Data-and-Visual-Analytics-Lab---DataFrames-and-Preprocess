#!/usr/bin/env python
# coding: utf-8

# ## BERCHMANS KEVIN 
# 
# 

# ## ``Department of Data Science - Data and Visual Analytics Lab``

# # ``Numpy & Pandas``

# ## `Import CSV file into DataFrames and Preprocess that dataset`

# # Numpy

# In[1]:


import numpy as np


# In[2]:


bk=np.genfromtxt("CA_01 Assignment-(215229107).csv",delimiter=',',dtype=str,skip_header=0)  


# In[3]:


bk


# #### Size

# In[4]:


np.shape(bk)


# #### Data rows here

# In[5]:


np.shape(bk)[0]


# #### Data columns here

# In[6]:


np.shape(bk)[1]


# #### Dimensions

# In[7]:


bk.ndim


# In[8]:


np.ndim(bk)


# #### Type

# In[9]:


type(bk)


# #### DataType

# In[10]:


bk.dtype


# #### Show top 6 rows

# In[11]:


bk[:5]


# #### Select first 3 items in 4th column

# In[12]:


bk[:3,3]


# #### Select first item in first all column

# In[13]:


bk[0,:]


# #### Show 3rd Row

# In[14]:


bk[2]


# #### Show 1st Column

# In[15]:


bk[:,0]


# #### Show 2nd Column

# In[16]:


bk[:,1]


# #### Show 3rd Column

# In[17]:


bk[:,2]


# #### Show 4th Column

# In[18]:


bk[:,3]


# #### Show 5th Column

# In[19]:


bk[:,4]


# #### Shoe 6th Column

# In[20]:


bk[:,5]


# #### Show top 1 to 14 rows

# In[21]:


bk[1:,:][:14,:]


# #### Select entire array

# In[22]:


bk[:]


# #### Iterating Arrays Using nditer()

# In[23]:


for q in np.nditer(bk):
    print(q)


# #### 1 Dimensional Numpy Arrays
#    #### Select 5th row all columns and show it's 4th value

# In[24]:


q1 = bk[4,:]


# In[25]:


q1


# In[26]:


q1[3]


# #### Convert to Integer values and Show

# In[27]:


q2 = bk[5,4].astype(int)


# In[28]:


q2


# #### Vectorization Operations

# #### Missing Values

# In[29]:


q4 = bk[3,:]

q5 = bk[4,:]


# #### Fill the missing values

# In[30]:


q6 = bk[3,4] = 90


# In[31]:


q6


# In[32]:


q7 = bk[4,-1] = 182


# In[33]:


q7


# #### Display the q4 qnd q5

# In[34]:


q4


# In[35]:


q5


# ##### Aggregate Functions

# In[36]:


bk[1:,3:].astype('int')


# In[37]:


bk[1:,3:].astype('int').max()


# In[38]:


bk[1:,3:].astype('int').min()


# In[39]:


bk[1:,3:].astype('int').mean()


# In[40]:


bk[1:,3:].astype('int').mean(axis=0)


# In[41]:


bk[1:,3:].astype('int').mean(axis=1).shape


# In[42]:


bk[1:,3:].astype('int').mean(axis=1)


# In[43]:


bk[1:,3:].astype('int').mean(axis=1).shape


# In[44]:


np.median(bk[1:,3:].astype('int'))


# In[45]:


bk[1:,3:].astype('int').std()


# #### Transpose

# In[46]:


bk[:,3:].transpose()


# In[47]:


np.transpose(bk).shape


# #### NumPy Array Comparisons 

# In[48]:


bk[1:,3:].astype(int) > 100


# In[49]:


bk[1:,3:].astype(int) > 55


# In[50]:


bk[1:,3:].astype(int) < 55


# In[51]:


bk[1:,3:].astype(int) < 100


# #### Convert wines data into 1D array 

# In[52]:


bk.ravel()


# In[53]:


bk.ravel().shape


# #### Sort Column Ascendind Order

# In[54]:


np.sort(bk[1:,3:])


# #### Make sorting to take place in-place 

# In[55]:


bk[1:,3:].sort()


# In[56]:


bk[1:,3:]


#   

# #### Save and Load 

# In[57]:


np.save('Berchmans_Numpy',bk)


# In[58]:


c = np.load('Berchmans_Numpy.npy')
c


# ## Pandas

# In[59]:


import pandas as pd


# In[60]:


bk_29 = pd.read_csv("CA_01 Assignment-(215229107).csv")
bk_29


# In[61]:


bk_29.head()


# In[62]:


bk_29.size


# In[63]:


bk_29.shape


# In[64]:


bk_29.columns


# In[65]:


bk_29.info


# In[66]:


bk_29.info()


# In[67]:


bk_29.describe()


# In[68]:


bk_29.isnull


# In[69]:


bk_29.isnull().sum()


# In[70]:


bk_29.isnull().values.sum()


# In[71]:


e = bk_29.bfill(axis=0)
e


# In[72]:


e.isnull().values.sum()


# ## Normalization

# In[73]:


from sklearn.preprocessing import MinMaxScaler

norm1 = MinMaxScaler()

norm1.fit_transform(e[['Height','Weight']])


# In[ ]:




