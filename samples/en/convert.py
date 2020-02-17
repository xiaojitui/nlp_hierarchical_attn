#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[3]:


files = os.listdir('./doc_level')
files[:10]


# In[ ]:





# In[30]:


from tqdm import tqdm
pos_doc = []
for file in tqdm(files, total = len(files)):
    with open('./doc_level/pos/' + file, 'rb') as f:
        text = f.read().decode()
    pos_doc.append(text)


# In[33]:


import pickle
with open('./doc_level/pos.pkl', 'wb') as f:
    pickle.dump(pos_doc, f)


# In[ ]:





# In[ ]:





# In[10]:


with open('./doc_level/434_4.txt', 'rb') as f:
    t1 = f.read().decode()
    
with open('./doc_level/434_4.txt', 'rb') as f:
    t2 = f.read()


# In[12]:


t1


# In[ ]:




