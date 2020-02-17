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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore Info and Warning
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

# In[ ]:





# In[ ]:





# ## load data

# In[2]:


with open('/ids/datasets/CNN_datasets/temp_data/all_cn_data.pkl', 'rb') as f:
    alldata = pickle.load(f)

#len(alldata) 


# In[3]:


#for year in alldata:
#    print(year, '\t', len(alldata[year]['X']), '\t', len(alldata[year]['y']))


# In[4]:



# In[5]:


X_train = []
y_train = []
X_test = []
y_test = []

for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016]:
    for i in range(len(alldata[year]['X'])):
        if alldata[year]['X'][i] != '':
            X_train.append(alldata[year]['X'][i])
            y_train.append(alldata[year]['y'][i])
    
for year in [2017, 2018]:
    for i in range(len(alldata[year]['X'])):
        if alldata[year]['X'][i] != '':
            X_test.append(alldata[year]['X'][i])
            y_test.append(alldata[year]['y'][i])
    
#len(X_train), len(X_test), len(y_train), len(y_test)
# (30840, 2272, 30840, 2272) is not consider ''


# In[ ]:





# In[ ]:





# In[ ]:





# ## preprocess

# In[6]:


with open('../model/contractions.pkl', 'rb') as f:
    contractions = pickle.load(f)


# In[91]:


# '一、'
def replace_digit(txt):
    digit_p = r'[0-9]+[,]*[，]*[.]*[%]*[％]*'
    txt_clean = re.sub(digit_p, 'digit_char', txt, flags=re.MULTILINE)
    #txt_clean = re.sub(r'\s*[digit_char]+\s*', 'digit_char', txt_clean, flags=re.MULTILINE)
    txt_clean = re.sub(r'\s*[digit_char]+\s*', '', txt_clean, flags=re.MULTILINE) #[元,吨]*
    return txt_clean



X_train_clean = []

for i in tqdm(range(len(X_train)), total = len(X_train)):
    x = X_train[i]
    x = replace_digit(x)
    x = utils.clean_sentence([x])
    x = utils.replace_contr(x, contractions)
    #x = utils.split_document(x[0])
    #x, _ = utils.refine_document(x) # use this if need to train "phrase"
    X_train_clean.append(x[0])
    
X_test_clean = []

for i in tqdm(range(len(X_test)), total = len(X_test)):
    x = X_test[i]
    x = replace_digit(x)
    x = utils.clean_sentence([x])
    x = utils.replace_contr(x, contractions)
    #x = utils.split_document(x[0])
    #x, _ = utils.refine_document(x) # use this if need to train "phrase"
    X_test_clean.append(x[0])
    
#len(X_train_clean), len(X_test_clean)


# In[ ]:





# In[ ]:





# In[ ]:





# ## prepare doc, sen, and study sample statistics

# In[49]:


# convert data to sentences, then phrases
X_train_doc = []
for x in tqdm(X_train_clean, total = len(X_train_clean)):
    doc = utils.split_document(x)
    X_train_doc.append(doc)

    
# refine , if necessary
#X_train_doc, _ = utils.refine_documemt(X_train_doc, contractions)


X_test_doc = []
for x in tqdm(X_test_clean, total = len(X_test_clean)):
    doc = utils.split_document(x)
    X_test_doc.append(doc)

    
#X_test_doc, _ = utils.refine_documemt(X_test_doc, contractions)

# refine , if necessary
print(len(X_train_clean), len(X_train_doc), len(X_test_clean), len(X_test_doc))





bert_path = '../bert/checkpoint/ch'
bert_config_file = '../bert/checkpoint/ch/bert_config.json'
bert_vocab_file = '../bert/checkpoint/ch/vocab.txt'
init_checkpoint = '../bert/checkpoint/ch/bert_model.ckpt'
graph_file = '../bert/tmp_ch/graph_doc_dt' # where to put temporary graph
model_dir = '../bert/tmp' # where to put temporary graph
select_layers = [-4, -3, -2, -1] 
#select_layers = [0] # get the cls token
doc_len = 250
sen_len = 150

batch_size = 32


# In[63]:


tokenizer, estimator = embedding_bert.prepare_bert(bert_vocab_file, bert_config_file, init_checkpoint, sen_len, 
                                                   select_layers, batch_size, graph_file, model_dir)


# In[64]:





##vocab_size = emb.shape[0]
#embedding_size = emb.shape[1]
#pretrained_emb = True
#finetune_emb = False

embedding_size = 3072
doc_len = 250
sen_len = 150
keep_prob = 0.5

epochs = 5 #20
batch_size = 16 #64

n_hidden = 50
n_layer = 2

atten_size = 50
l2reg = 0.0 #1e-5
use_mask = True
sen_CLS = True

####n_class = 2
lr = 1e-4

#savepath = './saved_weights/best_model.ckpt'
#finalpath = './saved_weights/final_model.ckpt'
savepath = '/ids/datasets/CNN_datasets/saved_weights_n/best_model.ckpt'
finalpath = '/ids/datasets/CNN_datasets/saved_weights_n/final_model.ckpt'


# In[ ]:





# In[100]:


tf.reset_default_graph()

with tf.name_scope('inputs'):
    
    #X = tf.placeholder(shape=(None, doc_len, sen_len), dtype=tf.int64, name='inputs') #len_seq = 40
    X_emb = tf.placeholder(shape=(None, doc_len, sen_len, embedding_size), dtype=tf.float32, name='inputs')
    y = tf.placeholder(shape=(None,), dtype=tf.float32, name='labels') #tf.int64
    is_training = tf.placeholder_with_default(False, shape = [], name='is_training')
    doc_seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='doc_seq_length')
    sen_seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='sen_seq_length')
    
## prepare embedding
#with tf.device('/cpu:0'):
'''
with tf.name_scope('embedding'):
    # no pretrained_emb
    if pretrained_emb is False:
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
                                trainable = True)
    # load pretrained_emb
    ## see the backup: how to deal with too large emb
    else:
        #embedding = tf.get_variable('embedding', [vocab_size, embedding_size], trainable=False)
        embedding = tf.get_variable('embedding', initializer = emb, trainable = finetune_emb)
    
    X_embed = tf.nn.embedding_lookup(embedding, X) # None, doc_s, sen_s, embed_s
'''

#with tf.device('/gpu:1'):
with tf.name_scope('sen_rnn'):
    X_embed_reshape = tf.reshape(X_emb, [-1, sen_len, embedding_size])
    sen_rnn_outputs, sen_rnn_states = rnn_layer.bi_rnn(X_embed_reshape, n_hidden = n_hidden, seq_len = sen_seq_length, n_layer = n_layer, 
                                                       is_train = is_training, keep_prob = keep_prob, scope = 'sen_rnn_block')

with tf.name_scope('sen_attn'):
    sen_atten_out, sen_atten_w = attn_layer.atten_layer_project(sen_rnn_outputs, atten_size, n_layer = n_layer, l2reg = l2reg, 
                                                           seq_len = sen_seq_length, use_mask = use_mask, sen_CLS = sen_CLS, 
                                                                scope = 'sen_attn_block')
    # Dropout
    #sen_atten_out_drop = tf.layers.dropout(sen_atten_out, rate = 1-0.5, training = is_training) # tf.nn.dropout


#with tf.name_scope('sen_stack'):
    #sen_outs = stack_layer(sen_atten_out, sen_atten_w, sen_rnn_states, X_embed_reshape, scope = 'sen_stack_block')
    #sen_outs = stack_layer(sen_atten_out, sen_rnn_states, scope = 'sen_stack_block')
    #sen_outs_drop = tf.layers.dropout(sen_outs, rate = 1-0.5, training = is_training)

#with tf.device('/gpu:2'):
with tf.name_scope('doc_rnn'):
    doc_inputs = tf.reshape(sen_atten_out, [-1, doc_len, sen_atten_out.shape[1]])
    #doc_inputs = tf.reshape(sen_outs, [-1, doc_size, sen_outs.shape[1]])
    doc_rnn_outputs, doc_rnn_states = rnn_layer.bi_rnn(doc_inputs, n_hidden = n_hidden, seq_len = doc_seq_length, n_layer = n_layer, 
                                                       is_train = is_training, 
                                                       keep_prob = keep_prob, scope = 'doc_rnn_block')

with tf.name_scope('doc_attn'):
    doc_atten_out, doc_atten_w = attn_layer.atten_layer_project(doc_rnn_outputs, atten_size, n_layer = n_layer, l2reg = l2reg, 
                                                           seq_len = doc_seq_length, use_mask = use_mask, scope = 'doc_attn_block')
    # Dropout
    #doc_atten_out_drop = tf.layers.dropout(doc_atten_out, rate = 1-0.5, training = is_training) # tf.nn.dropout


#with tf.name_scope('doc_stack'):
    #doc_outs = stack_layer(doc_atten_out, doc_atten_w, doc_rnn_states, doc_inputs, scope = 'doc_stack_block')
    #doc_outs = stack_layer(doc_atten_out, doc_rnn_states, scope = 'doc_stack_block')
    #doc_outs_drop = tf.layers.dropout(doc_outs, rate = 1-0.5, training = is_training)

#with tf.name_scope('attn_stack'):    
    #doc_emb = stack_attn(X_embed_reshape, sen_atten_w, doc_atten_w, scope = None) # [None, doc]


#with tf.device('/gpu:3'):
with tf.name_scope('final_out'):
    #doc_emb_dense = tf.layers.dense(doc_emb, dense_n, activation = None)
    #final_outputs = tf.concat([doc_outs, doc_emb], axis = -1)
    final_outputs = doc_atten_out
    final_outputs_dense = tf.layers.dense(final_outputs, 16, activation = tf.nn.relu)
    final_outputs_drop = tf.layers.dropout(final_outputs_dense, rate = 1-0.5, training = is_training)

with tf.name_scope('logits'):
    optimizer, logits, cost = model.reg_train_op(final_outputs_drop, labels = y , ac_fn = None, 
                                                                    lr = lr, l2reg = l2reg) 
    init, saver = model.initializer()

    
with tf.name_scope('prediction'):
    y_pred = model.reg_predict_op(logits) 


# In[101]:


print(X_emb, '\n', X_embed_reshape)
print('\n\n')
print(sen_atten_out, '\n', sen_atten_w, '\n')
print('\n\n')
print(doc_inputs, '\n')
print(doc_atten_out, '\n', doc_atten_w, '\n')
print('\n\n')
print(final_outputs, '\n', final_outputs_dense)
print('\n\n')
print(logits,'\n',  cost)
print('\n\n')
print(y_pred)


# In[ ]:





# In[ ]:





# In[28]:


loss_record = {}
#acc_record = {}
rank_record = {}

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    
    #saver.restore(sess, finalpath)
    min_loss = np.inf

    for epoch in range(epochs):
        t0 = time.time()
        
        loss_train = 0
        loss_test = 0
        #acc_train = 0
        #acc_test = 0
        rank_train = 0
        rank_test = 0

        #print("epoch: {}\t".format(epoch), end="")

        # training
        n_batch = len(X_train_doc)//batch_size
        #for _ in range(n_batch):
        for _ in tqdm(range(n_batch), total = n_batch):
            X_batch, y_batch = utils.get_batch_data(X_train_doc, y_train, batch_size)
            X_batch_emb = embedding_bert.get_batch_emb(X_batch, doc_len, sen_len, tokenizer, estimator)
            #X_batch_emb = X_batch_emb[:, :, 0, :]
            #X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            doc_seq_len, sen_seq_len = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            #batch_seq_len = np.array([list(x).index(0) + 1 for x in X_batch])  # actual lengths of sequences
            _, loss, y_pred_val = sess.run([optimizer, cost, y_pred],
                                          feed_dict={X_emb: X_batch_emb, y: y_batch, 
                                                     doc_seq_length: doc_seq_len, 
                                                     sen_seq_length: sen_seq_len, is_training: True}) #seq_length: batch_seq_len

            #acc_train += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_train += loss
            rank_val = spearmanr(np.squeeze(y_pred_val), y_batch)[0]
            rank_train += rank_val
        #acc_train /= n_batch
        loss_train /= n_batch
        rank_train /= n_batch

        #if (epoch + 1) % (epochs//10) == 0:
        #if (epoch + 0) % 10 == 0:
            #print('Epoch:', '%d' % (epoch + 0), 'cost =', '{:.6f}'.format(loss))

        # testing
        n_batch = len(X_test_doc)//batch_size
        #for i in range(n_batch):
        for i in tqdm(range(n_batch), total = n_batch):
            X_batch, y_batch = utils.get_batch_test(X_test_doc, y_test, i, batch_size)
            X_batch_emb = embedding_bert.get_batch_emb(X_batch, doc_len, sen_len, tokenizer, estimator)
            #X_batch_emb = X_batch_emb[:, :, 0, :]
            #X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            doc_seq_len, sen_seq_len = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            #batch_seq_len = np.array([list(x).index(0) + 1 for x in X_batch])
            loss, y_pred_val = sess.run([cost, y_pred],
                                          feed_dict={X_emb: X_batch_emb, y: y_batch, 
                                                     doc_seq_length: doc_seq_len, 
                                                     sen_seq_length: sen_seq_len, is_training: False}) #seq_length: batch_seq_len
            #acc_test += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_test += loss
            rank_val = spearmanr(np.squeeze(y_pred_val), y_batch)[0]
            rank_test += rank_val
        #acc_train /= n_batch
        loss_test /= n_batch
        rank_test /= n_batch
        
        #acc_test /= n_batch

        loss_record[epoch] = [loss_train, loss_test]
        #acc_record[epoch] = [acc_train, acc_test]
        rank_record[epoch] = [rank_train, rank_test]

        #if epoch ==0 or (epoch + 1) % 4 == 0:
        if epoch >= 0:
            print("epoch: {}\t".format(epoch), end="")
            print("loss: {:.5f}, val_loss: {:.5f}, train_rank: {:.5f}, test_rank: {:.5f}".format(
                loss_train, loss_test, rank_train, rank_test))
            print(round(time.time() - t0, 3))
            

        if loss_test < min_loss:
            saver.save(sess, savepath)
            min_loss = loss_test

    saver.save(sess,  finalpath)


# In[ ]:





# In[ ]:





# In[ ]:



