#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from tqdm import tqdm
from scipy.stats import spearmanr
import pickle
import time

import sys
sys.path.append("../")
sys.path.append("../embedding")
from embedding import embedding_bert
from model import utils, model, rnn_layer, attn_layer

#tf.enable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore Info and Warning
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()


# In[ ]:





# In[2]:



with open('/ids/datasets/CNN_datasets/temp_data/X_train_3year.pkl', 'rb') as f:
    X_train_doc = pickle.load(f)

with open('/ids/datasets/CNN_datasets/temp_data/y_train_3year.pkl', 'rb') as f:
    y_train = pickle.load(f)
    
with open('/ids/datasets/CNN_datasets/temp_data/X_test.pkl', 'rb') as f:
    X_test_doc = pickle.load(f)

with open('/ids/datasets/CNN_datasets/temp_data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
    
print(len(X_train_doc), len(y_train), len(X_test_doc), len(y_test))


train_data = []

for i in tqdm(range(len(X_train_doc)), total = len(X_train_doc)):
    x = X_train_doc[i]
    y = y_train[i]
    train_data.append([x, y])


# In[3]:


len(train_data)


# In[4]:


for i in range(20):
    print(len(train_data[i][0]), '\t', train_data[i][1])


# In[5]:


data_part = []
for i in range(20):
    X = train_data[i][0]
    y = train_data[i][1]
    data_part.append([X[:i+1], y])


# In[6]:


y_true = [k[1] for k in data_part]
y_true = np.array(y_true)
y_true


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


bert_path = '../bert/checkpoint/ch'
bert_config_file = '../bert/checkpoint/ch/bert_config.json'
bert_vocab_file = '../bert/checkpoint/ch/vocab.txt'
init_checkpoint = '../bert/checkpoint/ch/bert_model.ckpt'
graph_file = '../bert/tmp_ch/graph_doc_test' # where to put temporary graph
model_dir = '../bert/tmp' # where to put temporary graph
select_layers = [-4, -3, -2, -1] 
#select_layers = [0] # get the cls token
doc_len = 5
sen_len = 10

batch_size = 32


# In[63]:


tokenizer, estimator = embedding_bert.prepare_bert(bert_vocab_file, bert_config_file, init_checkpoint, sen_len, 
                                                   select_layers, batch_size, graph_file, model_dir)


# In[11]:


xx = embedding_bert.get_batch_emb([data_part[0][0]], doc_len, sen_len, tokenizer, estimator)


# In[12]:


xx.shape


# In[ ]:





# In[9]:


a, b = embedding_bert.get_batch_seq([data_part[0][0], data_part[1][0]], doc_len, sen_len, tokenizer, tol = 2)
a, b


# In[10]:


a[0], b[0]


# In[11]:


xx[0].shape


# In[24]:


embedding_bert.get_batch_seq([data_part[4][0]], doc_len, sen_len, tokenizer, tol = 2)


# In[22]:


[data_part[0][0]]


# In[ ]:





# In[ ]:





# In[8]:


def get_train_data(train_data, batch_size):

    def generator_train():
        for data in train_data:
            x, y = data[0], data[1]
            x_emb = embedding_bert.get_batch_emb([x], doc_len, sen_len, tokenizer, estimator)
            doc_seq_len, sen_seq_len = embedding_bert.get_batch_seq([x], doc_len, sen_len, tokenizer, tol = 2)
            yield x_emb[0], y, doc_seq_len, sen_seq_len

    # check output_types
    #train_ds = tf.data.Dataset.from_generator(generator_train_batch(generator_train, 3), output_types=(tf.float32, tf.float32))
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types = (tf.float32, tf.float32, tf.int64, tf.int64))
    #train_ds = train_ds.map(_map_fn_train) #, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(batch_size) #shuffle_buffer_size = 100
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size) #batch_size=1
    train_ds = train_ds.prefetch(1) #buffer_size=1
    return train_ds


# In[9]:


def get_train_data11(train_data):

    def generator_train():
        for data in train_data:
            x, y = data[0], data[1]
            #x_emb = embedding_bert.get_batch_emb(x, doc_len, sen_len, tokenizer, estimator)
            doc_seq_len, sen_seq_len = embedding_bert.get_batch_seq([x], doc_len, sen_len, tokenizer, tol = 2)
            yield doc_seq_len, sen_seq_len

    # check output_types
    #train_ds = tf.data.Dataset.from_generator(generator_train_batch(generator_train, 3), output_types=(tf.float32, tf.float32))
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types = (tf.int64, tf.int64))
    #train_ds = train_ds.map(_map_fn_train) #, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(100) #shuffle_buffer_size = 100
    
    train_ds = train_ds.batch(3) #batch_size=1
    train_ds = train_ds.prefetch(1) #buffer_size=1

    return train_ds


# In[ ]:





# In[10]:


tf.reset_default_graph()
train_ds = get_train_data(data_part, batch_size = 3) # should be [Batch * [X, y, doc_seq, sen_seq]]
iter_op = train_ds.make_initializable_iterator()
ele_op = iter_op.get_next()
#train_ds = get_train_data11(data_part, 5, 3)


# In[ ]:





# In[11]:


np.array(y_true)


# In[12]:


config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    #sess.run(init)
    #saver.restore(sess, savepath)
    
    sess.run(iter_op.initializer)
    for i in range(6):
        print(i)
        x_batch, y_batch, c, d = sess.run(ele_op)
        c, d = c.flatten(), d.flatten()
        print(y_batch)


# In[13]:


x_batch.shape, y_batch.shape


# In[14]:


c


# In[15]:


d


# In[36]:


c.flatten(), d.flatten()


# In[41]:


c


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## backup

# In[ ]:


def gen():
    for i in range(1, 5):
        yield [i] * i

def get_batch_gen(gen, batch_size=2):
    def batch_gen():
        buff = []
        for i, x in enumerate(gen()):
            if i % batch_size == 0 and buff:
                yield np.concatenate(buff, axis=0)
                buff = []
            buff += [x]
            #buff.append(x)

        if buff:
            yield np.concatenate(buff, axis=0)

    return batch_gen

# Create dataset from generator
batch_size = 2
dataset = tf.data.Dataset.from_generator(get_batch_gen(gen, batch_size),
                                         (tf.int64)) #, tf.TensorShape([None, None]))

# Create iterator from dataset
iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()  # shape (None,)


with tf.Session() as sess:
    for i in range(2):
        print(sess.run(x))


# In[ ]:





# In[ ]:



def get_train_data(train_data):

    def generator_train():
        for data in train_data:
            yield data


    def generator_train_batch(generator_train, batch_size=10):
        def batch_gen():
            buff_x = []
            buff_y = []
            for i, (x, y) in enumerate(generator_train()):
                if i % batch_size == 0 and buff_x:
                    yield buff_x # np.concatenate(buff_x, axis=0)
                    buff_x = []
                    buff_y = []
                buff_x += [x[0]]
                buff_y += [y]

            if buff_x:
                yield buff_x

        return batch_gen
    
    # check output_types
    train_ds = tf.data.Dataset.from_generator(generator_train_batch(generator_train, 10), output_types=tf.string)
    #train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
    #train_ds = train_ds.map(_map_fn_train) #, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(100) #shuffle_buffer_size = 100
    
    #train_ds = train_ds.batch(10) #batch_size=1
    train_ds = train_ds.prefetch(1) #buffer_size=1

    return generator_train_batch #train_ds

