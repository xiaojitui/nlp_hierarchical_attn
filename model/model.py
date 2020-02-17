#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import os
#import re


# In[ ]:





# In[ ]:





# ## clf

# In[4]:


def clf_train_op(final_outputs, labels, ac_fn = None, lr = 0.001, l2reg = 0.0, n_class = 2):
    logits = tf.layers.dense(final_outputs, n_class, activation = ac_fn,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    base_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) ############get it
    cost = tf.add_n([base_cost] + reg_losses) 
    
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    
    Y_proba = tf.nn.softmax(logits)
    Y_pred = tf.argmax(Y_proba, axis = 1)
    
    #correct = tf.nn.in_top_k(targets = labels, predictions = logits, k = 1)
    correct = tf.equal(Y_pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return optimizer, logits, cost, accuracy, Y_proba


# In[5]:


def clf_predict_op(X_pred, Y_proba):
    Y_proba_val = Y_proba.eval(feed_dict={X: X_pred})
    y_pred = np.argmax(Y_proba_val, axis=1)
    
    return y_pred


# In[ ]:


def clf_predict_op(X_pred, embed, Y_proba):
    Y_proba_val = Y_proba.eval(feed_dict={X: X_pred, embedding: embed})
    y_pred = np.argmax(Y_proba_val, axis=1)
    
    return y_pred


# In[ ]:





# ## reg

# In[6]:


def reg_train_op(final_outputs, labels, ac_fn = None, lr = 0.01, l2reg = 0.0):
    
    #outputs_1 = tf.layers.dense(final_outputs, 64, activation = ac_fn,
                             #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    
    #outputs_2 = tf.layers.dense(outputs_1, 8, activation = ac_fn,
                             #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    
    #outputs_1_drop = tf.layers.dropout(outputs_1, rate = 1-0.5, training = is_train) 
    logits = tf.layers.dense(final_outputs, 1, activation = None,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    
    logits_c = tf.clip_by_value(logits, -1.0, 1.0)
    labels_c = tf.clip_by_value(labels, -1.0, 1.0)
    base_cost = tf.reduce_mean(tf.square(logits_c - labels_c))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) ############get it
    cost = tf.add_n([base_cost] + reg_losses) 
        
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    
    return optimizer, logits, cost


# In[7]:


def reg_predict_op(logits):
    
    logits_c = tf.clip_by_value(logits, -1.0, 1.0)
    logits_c = tf.squeeze(logits_c, -1)
    return logits_c


# In[ ]:





# In[ ]:





# ## init

# In[8]:


def initializer():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() #
    
    return init, saver


# In[ ]:





# In[ ]:





# ## corr

# In[9]:


def my_func(x):
    x0 = np.array(x[0], dtype=np.float32) 
    x1 = np.array(x[1], dtype=np.float32)
    x1 = np.clip(x1, -1.0, 1.0)
    corr, p = spearmanr(x0, x1) 
    return np.array(corr, dtype=np.float32)

def my_corr(rank_pair): 
    rank_cor = tf.py_func(my_func, [rank_pair], tf.float32) 
    return rank_cor


# In[ ]:





# In[ ]:





# ## deal with layers

# In[11]:


# generate multiple dense layers
def dense_layer(inputs, n_layers, n_units, ac_fn = None, l2reg = 0.0):
    
    dense_layers = []
    
    for i in range(n_layers):
        if i == 0:
            layer_input = inputs
        else:
            layer_input = dense_layers[i-1]
            
        layer_output = tf.layers.dense(layer_input, n_units[i], activation = ac_fn, 
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg), 
                                  name='dense_layer_%s' % i)
        dense_layers.append(layer_output)
    
    return layer_output[-1]


# In[ ]:





# In[12]:


# stack attn_outputs + (rnn_states) + output_emb
def stack_layer(attn_outputs, attn_w, rnn_states, x_emb, scope = None):

    with tf.variable_scope(scope or 'stack_layer') as scope:
        if type(rnn_states) == tuple:
            rnn_states = tf.concat(rnn_states[-1][-1], -1)  # [None, n_hidden*2]
        else:
            rnn_states = rnn_states   # [None, n_hidden]
            
            
        ## add weighted "sentence_emb" or "doc_emb"
        # attn_w = [None, seq]
        # x_emb = [None, seq, embedding]
        attn_w_1 = tf.expand_dims(attn_w, -1)
        output_emb = tf.reduce_sum(attn_w_1 * x_emb, axis = 1) # [None, embedding]
        
        
        ## add weighted "rnn_state"
        #attn_w_2 = tf.reduce_mean(attn_w, axis = 1) #[None,]
        #output_rs = attn_w_2 * rnn_states # [None, n_hidden]
        
        final_output = tf.concat([attn_outputs, rnn_states, output_emb], axis = -1)
        
    return final_output


# In[ ]:





# In[13]:


# generate doc emb
def get_doc_emb(x_emb, sen_attn_w, doc_attn_w, scope = None):
    
    with tf.variable_scope(scope or 'stack_attn') as scope:
        ## add weighted "sentence_emb" and "doc_emb"
        # sen_attn_w = [None*doc, sen]
        # x_emb = [None*doc, sen, embedding]
        attn_w_1 = tf.expand_dims(sen_attn_w, -1) #[None*doc, sen, 1]
        sen_emb = tf.reduce_sum(attn_w_1 * x_emb, axis = 1) # [None*doc, embedding]
        
        sen_emb_reshape = tf.reshape(sen_emb, [-1, doc_size, emb_size]) # [None, doc, embedding]
        # doc_attn_w = [None, doc]
        attn_w_2 = tf.expand_dims(doc_attn_w, -1) #[None, doc, 1]
        doc_emb = tf.reduce_sum(attn_w_2 * sen_emb_reshape, axis = 1) # [None, embedding]
        
    return doc_emb


# In[ ]:





# In[ ]:





# In[ ]:




