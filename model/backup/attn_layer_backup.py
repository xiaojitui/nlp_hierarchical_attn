#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[ ]:





# ## atten1: self

# In[7]:


def atten_self(outputs, final_states, n_layer = None, scope = None):
    # Self dot Attention: (output.T * softmat(output*state)) * out
    
    
    with tf.variable_scope(scope or 'attention'):
        # if outputs are bi-rnn
        if type(outputs) == tuple:
            #out_w = tf.Variable(tf.random_normal([self.n_hidden * 2, self.n_class]))
            output = tf.concat([outputs[0], outputs[1]], 2)  # [None, len_seq, n_hidden*2]

            # concat h state
            if n_layer is not None:
                final_hidden_state = tf.concat([final_states[0][-1][1], final_states[1][-1][1]], 1) # [None, n_hidden*2]
            else:
                final_hidden_state = tf.concat([final_states[0][1], final_states[1][1]], 1) # [None, n_hidden*2]
        # if outputs are rnn
        else:
            #out_w = tf.Variable(tf.random_normal([self.n_hidden, self.n_class]))
            output = outputs

            # concat h state
            if n_layer is not None:
                final_hidden_state = final_states[-1][1] # [None, n_hidden]
            else:
                final_hidden_state = final_states[1]


        final_hidden_state = tf.expand_dims(final_hidden_state, 2) 
        # final_hidden_state : [None, n_hidden*2, 1] if bi-rnn or [None, n_hidden, 1] if rnn

        attn_weights = tf.squeeze(tf.matmul(output, final_hidden_state), 2) # attn_weights : [None, len_seq]
        soft_attn_weights = tf.nn.softmax(attn_weights, 1) # [None, len_seq]


        context = tf.matmul(tf.transpose(output, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2)) 
        # tf.transpose(output, [0, 2, 1]): [None, n_hidden*2, len_seq] if bi-rnn or [None, n_hidden, len_seq] if rnn
        # tf.expand_dims(soft_attn_weights, 2): [None, len_seq, 1]
        # context : [None, n_hidden * 2, 1]
        atten_out = tf.squeeze(context, 2) # [None, n_hidden * 2] if bi-rnn or [None, n_hidden] if rnn

        #atten_final_out = tf.matmul(context, out_w)
        # atten_finalout = [None, n_class]

    return atten_out, soft_attn_weights


# In[ ]:





# ## atten2: project

# In[9]:


def atten_layer_project(outputs, atten_size, n_layer = None, l2reg = 0.0, seq_len = None, use_mask = False, scope = None):
    # atten_layer: 
    # (1) output_projection = fully_connect(output, atten_size, tanh)
    # (2) output.T * softmat(output_projection*atten_vector)
    # tanh(fully_connect(outputs to atten_size))
    # outputs: [None, seq, hidden]
    # atten_vect: [atten_size, ]

    with tf.variable_scope(scope or 'attention') as scope:
        # if outputs are bi-rnn
        if type(outputs) == tuple:
            output = tf.concat([outputs[0], outputs[1]], 2)  # [None, len_seq, n_hidden*2]
        # if outputs are rnn
        else:
            output = outputs   # [None, len_seq, n_hidden]

        #with tf.variable_scope(scope or 'attention') as scope:

        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                             shape=[atten_size],
                                             regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg),
                                             dtype=tf.float32)
        input_projection = tf.layers.dense(output, atten_size,
                                            activation=tf.tanh,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
        attn_weights = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keepdims=True)
        # input_projection: [None, seq, atten]
        # attention_context_vector: [atten, ]
        # tf.multiply(input_projection, attention_context_vector): [None, seq, atten]
        # attn_weights: [None, seq, 1]

        soft_attn_weights = tf.nn.softmax(attn_weights, 1) # [None, seq, 1]

        if use_mask:
            soft_attn_weights = tf.squeeze(soft_attn_weights, 2) # [None, seq]
            max_len = soft_attn_weights.shape[1].value
            mask = tf.sequence_mask(seq_len, max_len, dtype = tf.float32)

            mask_attn = mask * soft_attn_weights
            norm_mask = tf.reduce_sum(mask_attn, -1, keepdims = True)
            norm_mask = tf.where(tf.not_equal(norm_mask, 0), norm_mask, tf.ones_like(norm_mask)) # cannot divide by 0
            mask_attn = tf.divide(mask_attn, norm_mask)

            soft_attn_weights = tf.expand_dims(mask_attn, 2)

        context = tf.matmul(tf.transpose(output, [0, 2, 1]), soft_attn_weights)
        # [None, n_hidden * 2, 1] if bi-rnn or [None, n_hidden, 1] if rnn

        atten_out = tf.squeeze(context, 2)
        soft_attn_weights = tf.squeeze(soft_attn_weights, 2)

    return atten_out, soft_attn_weights


# In[ ]:





# In[ ]:





# ## atten3: weight parameters

# In[11]:


def atten_layer_weight(outputs, atten_size, n_layer = None, l2reg = 0.0, seq_len = None, use_mask = False, scope = None):
    # atten_layer: 
    # (1) output_projection = tanh(output*w + b) * u
    # (2) output.T * softmat(output_projection)
    # outputs: [None, seq, hidden]
    # atten_vect: [atten_size, ]
    with tf.variable_scope(scope or 'attention') as scope:
        # if outputs are bi-rnn
        if type(outputs) == tuple:
            output = tf.concat([outputs[0], outputs[1]], 2)  # [None, len_seq, n_hidden*2]
        # if outputs are rnn
        else:
            output = outputs   # [None, len_seq, n_hidden]

        hidden_size = output.shape[2].value 


        # Trainable parameters
        w = tf.Variable(tf.random_normal([hidden_size, atten_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([atten_size], stddev=0.1))
        u = tf.Variable(tf.random_normal([atten_size], stddev=0.1))

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        # If axes is a scalar, sum over the last N axes of a and the first N axes of b in order.
        v = tf.tanh(tf.tensordot(output, w, axes=1) + b)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u, axes=1)  # (B,T) shape
        attn_weights = tf.expand_dims(vu, axis = 2)
        soft_attn_weights = tf.nn.softmax(attn_weights, 1)         # (B,T, 1) shape

        if use_mask:
            soft_attn_weights = tf.squeeze(soft_attn_weights, 2)
            max_len = soft_attn_weights.shape[1].value
            mask = tf.sequence_mask(seq_len, max_len, dtype = tf.float32)

            mask_attn = mask * soft_attn_weights
            norm_mask = tf.reduce_sum(mask_attn, -1, keepdims = True)
            norm_mask = tf.where(tf.not_equal(norm_mask, 0), norm_mask, tf.ones_like(norm_mask)) # cannot divide by 0
            mask_attn = tf.divide(mask_attn, norm_mask)

            soft_attn_weights = tf.expand_dims(mask_attn, 2)

        # Output of (Bi-)RNN is reduced with attention vector;
        context = tf.matmul(tf.transpose(output, [0, 2, 1]), soft_attn_weights)
        # [None, n_hidden * 2, 1] if bi-rnn or [None, n_hidden, 1] if rnn

        atten_out = tf.squeeze(context, 2)
        soft_attn_weights = tf.squeeze(soft_attn_weights, 2)
        # the result has (B,D) shape

    return atten_out, soft_attn_weights


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## backup

# In[12]:


# use tensorflow's attention wrap functions
def atten_wrap():
     pass


# In[ ]:





# In[ ]:





# In[ ]:




