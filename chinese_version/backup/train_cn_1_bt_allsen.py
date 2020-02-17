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


#os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"

# In[ ]:





# In[ ]:





# ## load data

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





bert_path = '../bert/checkpoint/ch'
bert_config_file = '../bert/checkpoint/ch/bert_config.json'
bert_vocab_file = '../bert/checkpoint/ch/vocab.txt'
init_checkpoint = '../bert/checkpoint/ch/bert_model.ckpt'
graph_file = '../bert/tmp_ch/graph_doc_dt' # where to put temporary graph
model_dir = '../bert/tmp' # where to put temporary graph
select_layers = [-4, -3, -2, -1] 
#select_layers = [0] # get the cls token
doc_len = 100
sen_len = 50

batch_size = 32


# In[40]:


tokenizer, estimator = embedding_bert.prepare_bert(bert_vocab_file, bert_config_file, init_checkpoint, sen_len, 
                                                   select_layers, batch_size, graph_file, model_dir)




# ## Train section

# In[35]:


#vocab_size = emb.shape[0]
#embedding_size = emb.shape[1]
#pretrained_emb = True
#finetune_emb = False

embedding_size = 3072
doc_len = 100
sen_len = 50
keep_prob = 0.5

epochs = 5 #20
batch_size = 16 #64

n_hidden = 50
n_layer = 2

atten_size = 50
l2reg = 0.0 #1e-5
use_mask = True
sen_CLS = False

##n_class = 2
lr = 1e-4

#savepath = './saved_weights/best_model.ckpt'
#finalpath = './saved_weights/final_model.ckpt'
savepath = '/ids/datasets/CNN_datasets/saved_weights_n/best_model.ckpt'
finalpath = '/ids/datasets/CNN_datasets/saved_weights_n/final_model.ckpt'


# In[36]:


# generate doc emb
def get_doc_emb(x_emb, doc_attn_w, scope = None):
    
    with tf.variable_scope(scope or 'stack_attn') as scope:
        # X_emb = [None, doc, embedding]
        # doc_attn_w = [None, doc]
        doc_attn_w = tf.expand_dims(doc_attn_w, -1) #[None, doc, 1]
        doc_emb = tf.reduce_sum(doc_attn_w * X_emb, axis = 1) # [None, embedding]
        
    return doc_emb


# In[37]:


tf.reset_default_graph()

with tf.name_scope('inputs'):
    
    ##X = tf.placeholder(shape=(None, sen_len), dtype=tf.int64, name='inputs')
    X_emb = tf.placeholder(shape=(None, doc_len, 3072), dtype=tf.float32, name='inputs')
    y = tf.placeholder(shape=(None,), dtype=tf.float32, name='labels') #tf.int64
    is_training = tf.placeholder_with_default(False, shape = [], name='is_training')
    seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='seq_length')
    
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
#with tf.device('/gpu:3'):
with tf.name_scope('rnn_layer'):
    rnn_outputs, rnn_states = rnn_layer.bi_rnn(X_emb, n_hidden = n_hidden, seq_len = seq_length, n_layer = n_layer, is_train = is_training, 
                                               keep_prob = keep_prob) #### need seq_length??

with tf.name_scope('attention_layer'):
    atten_out, soft_atten_weights = attn_layer.atten_layer_project(rnn_outputs, atten_size, n_layer = n_layer, 
                                                                   l2reg = l2reg, seq_len = seq_length, use_mask = use_mask, 
                                                                   sen_CLS = sen_CLS)
    # Dropout
    #atten_out_drop = tf.nn.dropout(atten_out, keep_prob)

with tf.name_scope('doc_layer'):
    doc_emb = get_doc_emb(X_emb, soft_atten_weights)
    #doc_out = tf.concat([doc_emb, atten_out], axis = -1)
    doc_out = doc_emb
    doc_out_1 = tf.layers.dense(doc_out, 84, activation = tf.nn.relu)
    doc_out_drop = tf.layers.dropout(doc_out_1, keep_prob, training = is_training) #, is_train = is_training)

with tf.name_scope('logits'):
    optimizer, logits, cost = model.reg_train_op(doc_out_drop, labels = y , ac_fn = None, 
                                                                    lr = lr, l2reg = l2reg)  
    init, saver = model.initializer()


with tf.name_scope('prediction'):
    y_pred = model.reg_predict_op(logits) 


# In[ ]:





# In[ ]:





# In[38]:


print(X_emb)
print('\n\n')
print(atten_out, '\n', soft_atten_weights)
print('\n\n')
print(doc_emb, '\n', doc_out, '\n', doc_out_1)
print('\n\n')
print(logits,'\n',  cost)
print('\n\n')
print(y_pred)


# In[ ]:





# In[ ]:





# In[ ]:


loss_record = {}
#acc_record = {}
rank_record = {}

config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
#with tf.Session() as sess:
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
        for n_iter in tqdm(range(n_batch), total = n_batch):
            X_batch, y_batch = utils.get_batch_data(X_train_doc, y_train, batch_size)
            X_batch_emb = embedding_bert.get_batch_emb(X_batch, doc_len, sen_len, tokenizer, estimator)
            X_batch_emb = X_batch_emb[:, :, 0, :]
            X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            _, loss, y_pred_val = sess.run([optimizer, cost, y_pred],
                                          feed_dict={X_emb: X_batch_emb, y: y_batch, seq_length: X_batch_seq, is_training:True}) #seq_length: batch_seq_len

            #acc_train += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_train += loss
            try:
                rank_val = spearmanr(np.squeeze(y_pred_val), y_batch)[0]
            except:
                rank_val = 0
            
            rank_train += rank_val
            if n_iter % 20 == 0:
                tqdm.write(str(rank_train/(n_iter + 1))) #, flush = True
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
            X_batch_emb = X_batch_emb[:, :, 0, :]
            X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            loss, y_pred_val = sess.run([cost, y_pred],
                                          feed_dict={X_emb: X_batch_emb, y: y_batch, seq_length: X_batch_seq, is_training:False}) #seq_length: batch_seq_len
            #acc_test += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_test += loss
            try:
                rank_val = spearmanr(np.squeeze(y_pred_val), y_batch)[0]
            except:
                rank_val = 0
            rank_test += rank_val
        #acc_train /= n_batch
        loss_test /= n_batch
        rank_test /= n_batch

        loss_record[epoch] = [loss_train, loss_test]
        #acc_record[epoch] = [acc_train, acc_test]
        rank_record[epoch] = [rank_train, rank_test]

        #if epoch ==0 or (epoch + 1) % 4 == 0:
        if epoch >= 0:
            print("epoch: {}\t".format(epoch), end="")
            print("loss: {:.5f}, val_loss: {:.5f}, train_rank: {:.5f}, test_rank: {:.5f}".format(
                loss_train, loss_test, rank_train, rank_test), end = '\t')
            print(round(time.time() - t0, 3))
            


        if loss_test < min_loss:
            saver.save(sess, savepath)
            min_loss = loss_test

    saver.save(sess,  finalpath)


# In[ ]:




