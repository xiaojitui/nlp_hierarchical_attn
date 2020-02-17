#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[ ]:





# ## rnn

# In[2]:


def rnn(inputs, n_hidden, seq_len = None, n_layer = None, is_train = False, keep_prob = 1.0, scope = None):
    
    with tf.variable_scope(scope or 'rnn'):
        if is_train is False:
            keep_prob = 1.0
    
        if n_layer is not None: 
            cells = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = n_hidden),input_keep_prob = keep_prob) 
                     for _ in range(n_layer)]
            stacked = tf.contrib.rnn.MultiRNNCell(cells)
            stacked = tf.nn.rnn_cell.DropoutWrapper(stacked, output_keep_prob = keep_prob)
            outputs, final_states = tf.nn.dynamic_rnn(cell=stacked, inputs=inputs, sequence_length = seq_len, 
                                                      dtype=tf.float32)
            #[None, seq, hidden]
            #(layer1, layer2...), each is (c = [None, hidden],  h = [None, hidden])
        else:
            cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = n_hidden), input_keep_prob = keep_prob, 
                                                 output_keep_prob = keep_prob)

            outputs, final_states = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length = seq_len, 
                                                      dtype=tf.float32)
            #[None, seq, hidden]
            #(c = [None, hidden],  h = [None, hidden])

    return outputs, final_states 


# In[ ]:





# ## bi_rnn

# In[5]:


def bi_rnn(inputs, n_hidden, seq_len = None, n_layer = None, is_train = False, keep_prob = 1.0, scope = None):
    
    with tf.variable_scope(scope or 'bi_rnn'):
        if is_train is False:
            keep_prob = 1.0
    
        if n_layer is not None: 
            cell_1 = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = n_hidden),input_keep_prob = keep_prob) 
                      for _ in range(n_layer)]
            cell_2 = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = n_hidden),input_keep_prob = keep_prob) 
                      for _ in range(n_layer)]

            stacked_1 = tf.contrib.rnn.MultiRNNCell(cell_1)
            stacked_1 = tf.nn.rnn_cell.DropoutWrapper(stacked_1, output_keep_prob = keep_prob)
            stacked_2 = tf.contrib.rnn.MultiRNNCell(cell_2)
            stacked_2 = tf.nn.rnn_cell.DropoutWrapper(stacked_2, output_keep_prob = keep_prob)

            # bidirectional RNN
            # ((fw, bw), (fw_s, bw_s))
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked_1, cell_bw=stacked_2,
                                                                    inputs=inputs, sequence_length = seq_len, dtype=tf.float32)
            # scope = scope
            #(fw = [None, seq, hidden], bw = [None, seq, hidden])
            #(fw_s, bw_s), each is (layer1, layer2..), each is (c = [None, hidden],  h = [None, hidden])

        else:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = n_hidden), 
                                                    input_keep_prob = keep_prob, output_keep_prob = keep_prob)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = n_hidden), 
                                                    input_keep_prob = keep_prob, output_keep_prob = keep_prob)

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, 
                                                                    inputs=inputs, sequence_length = seq_len, dtype=tf.float32)
            # scope = scope
            #(fw = [None, seq, hidden], bw = [None, seq, hidden])
            #(fw_s, bw_s), each is (c = [None, hidden],  h = [None, hidden])

    return outputs, final_states


# In[ ]:





# In[ ]:




