#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[ ]:





# In[ ]:


# mask = tf.sequence_mask(seq_len, max_len, dtype = tf.float32)


# ## test section

# In[21]:


tf.reset_default_graph()
sq_lengths = tf.placeholder(shape=(None,), dtype=tf.int64)
mask = tf.sequence_mask(sq_lengths, 6, dtype = tf.float32)

paddings = tf.constant([[0, 0,], [1, 0]])
mask_add = tf.pad(tf.sequence_mask(sq_lengths - 2, 6 - 1, dtype = tf.float32), paddings, 'CONSTANT')
mask, mask_add


# In[ ]:





# In[23]:


seq_test = np.array([3, 4, 6, 6, 5])
with tf.Session() as sess:
    mask_out =sess.run(mask, feed_dict = {sq_lengths: seq_test})
    mask_out_1 =sess.run(mask_add, feed_dict = {sq_lengths: seq_test})
print(mask_out.shape, mask_out_1.shape)    
print(mask_out)
print(mask_out_1)


# In[ ]:





# ### if want to add 0 for 1st element

# In[41]:


add_col = np.reshape([0]*mask_out.shape[0], (-1, 1))
np.concatenate((add_col, mask_out), axis = 1)


# In[ ]:





# In[ ]:





# In[ ]:





# ### if want to add 0 for last element

# In[42]:


# minus 1 for the seq_test
seq_test = np.array([2, 4, 1, 7, 3])
with tf.Session() as sess:
    mask_out =sess.run(mask, feed_dict = {sq_lengths: seq_test - 1})
    
print(mask_out)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




