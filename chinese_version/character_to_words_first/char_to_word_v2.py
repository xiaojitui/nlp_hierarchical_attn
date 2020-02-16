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
import jieba as J
from scipy.stats import spearmanr



def read_config(path):
    config = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip() != '':
                if line.split('=')[0].strip().startswith('#'):
                    continue
                field = line.split('=')[0].strip()
                val = line.split('=')[-1].strip()
                
                try:
                    val = int(val)
                except:
                    try:
                        val = float(val)
                    except:
                        pass
                config[field] = val
    return config



def loaddata(path):
    data = pd.read_csv(path)
    return data



def get_train_test(X, y, train_ratio = 0.9):
    
    idx = np.arange(len(X)) 
    np.random.shuffle(idx) 
    idx_1 = idx[:int(train_ratio*len(X))] 
    idx_2 = idx[int(train_ratio*len(X)):]
    
    X_train = np.array([X[k] for k in idx_1])
    y_train = np.array([y[k] for k in idx_1])
    X_test = np.array([X[k] for k in idx_2])
    y_test = np.array([y[k] for k in idx_2])
    
    return X_train, y_train, X_test, y_test
    
    

def get_sen_stat(sentences, cut_thres = 0.90):
    
    sen_len = []
    
    for sen in sentences:
        sen_len.append(len(sen.split()))
    
    cut_val = -1
    for i in range(max(sen_len)):
        if np.sum([k<=i for k in sen_len]) >= cut_thres * len(sen_len):
            cut_val = i
            break
            
    return cut_val, sen_len


def get_doc_stat(docs, doc_cut_thres = 0.90, sen_cut_thres = 0.90):
    doc_len = []
    sen_len = []
    
    for doc in docs:
        doc_len.append(len(doc))
        
        for sen in doc:
            sen_len.append(len(sen.split()))
            
    doc_cut_val = -1
    for i in range(max(doc_len)):
        if np.sum([k<=i for k in doc_len]) >= doc_cut_thres * len(doc_len):
            doc_cut_val = i
            break
            
    sen_cut_val = -1
    for i in range(max(sen_len)):
        if np.sum([k<=i for k in sen_len]) >= sen_cut_thres * len(sen_len):
            sen_cut_val = i
            break
       
    return doc_cut_val, sen_cut_val, doc_len, sen_len

def build_voc(sentences):
    words = ' '.join(sentences).split()
    words = [k.lower() for k in words]
    voc = list(set(words))
    
    return voc



def token(voc):
    word_to_idx = {w:i for i, w in enumerate(voc)}
    word_to_idx['UNK'] = len(voc)
    word_to_idx['PAD'] = len(voc) + 1
    idx_to_word = {i:w for w, i in word_to_idx.items()}
    
    return word_to_idx, idx_to_word


def split_sentence(txt):
    sents = re.split(r'\n|\s|;|；|。|，|\.|,|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|、|╱|∥', txt)
    sents = [c for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
    sents = list(filter(None, sents))
    return sents
 
def token_sen(sentence, sentence_size, word_to_idx):
    #words = sentence.split()[:sentence_size]
    words = split_sentence(sentence)
    words = words[:sentence_size]
    words = [k.lower() for k in words]

    if len(words) < sentence_size:
        words.extend(['PAD']*(sentence_size-len(words)))

    sen_idx = []
    for word in words:
        if word in word_to_idx:
            sen_idx.append(word_to_idx[word])
        else:
            sen_idx.append(word_to_idx['UNK'])

    return np.array(sen_idx)




def prepare(sentences, labels, sentence_size, word_to_idx):
    input_batch = []
    for sen in sentences:
        input_batch.append(token_sen(sen, sentence_size, word_to_idx))

    target_batch = []
    for out in labels:
        target_batch.append(out)

    return input_batch, target_batch



def prepare_batch(self, docs, labels):
    X_batch = []
    y_batch = []
    for doc in docs:
        doc_x = []
        doc_y = []
        for i in range(doc):
            sen = doc[i]
            sen_inputs = Dataset(parameters).token_sen(sen)
            doc_x.append(sen_inputs)
            doc_y.append(label[i])
        X_batch.append(doc_x)
        y_batch.append(doc_y)

    X_batch = np.stack(X_batch, axis = 0) # None,doc_len, sen_len
    y_batch = np.stack(y_batch, axis = 0) # None,doc_len, sen_len
    y_batch = y_batch[:, :, np.newaxis]
    return X_batch, y_batch 




def token_batch(sentences, sentence_size, word_to_idx):
    sen_idx_bath = []
    for sen in sentences:
        sen_idx = token_sen(sen, sentence_size, word_to_idx)
        sen_idx_bath.append(sen_idx)
        
    return np.array(sen_idx_bath)



def token_doc(docs, doc_size, sen_size, word_to_idx):
    doc_idx_batch = []
    
    for doc in docs:
        doc_idx = []
        for sen in doc:
            sen_idx = token_sen(sen, sen_size, word_to_idx)
            doc_idx.append([k for k in sen_idx])
           
        doc_idx = doc_idx[:doc_size]
        
        if len(doc_idx) < doc_size:
            doc_pad = [[word_to_idx['PAD']] * sen_size] * (doc_size-len(doc_idx))
            doc_idx.extend(doc_pad)
            
        doc_idx_batch.append(np.array(doc_idx))
        
    doc_idx_batch = np.stack(doc_idx_batch, axis=0)
        
    return doc_idx_batch




def get_batch_data(X, y, batch_size):
    
    idx = np.random.choice(len(X), batch_size, replace=False)
    y = np.array(y)
    X_batch = X[idx]
    y_batch = y[idx]
    
    return X_batch, y_batch


# In[ ]:





# In[ ]:





# ## emb + 2

# In[1]:


import pickle
from tqdm import tqdm
#with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2013_2018.pkl', 'rb') as f:
with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2010_2015.pkl', 'rb') as f:
    train_data = pickle.load(f)
        
#with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2010_2012.pkl', 'rb') as f:
with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2016_2018.pkl', 'rb') as f:
    test_data = pickle.load(f)
    
    
X_train = []
for doc in tqdm(train_data['x'], total = len(train_data['x'])):
    cur_doc = []
    for sen in doc:
        sen = ''.join(sen)
        cur_doc.append(sen)
    X_train.append(cur_doc)
y_train = train_data['y']


X_test = []
for doc in tqdm(test_data['x'], total = len(test_data['x'])):
    cur_doc = []
    for sen in doc:
        sen = ''.join(sen)
        cur_doc.append(sen)
    X_test.append(cur_doc)
y_test = test_data['y']


# In[5]:


print(len(X_train), len(y_train), len(X_test), len(y_test))


# In[10]:


# 0-1030: 2016, 1031-1904: 2017, 1905-2844: 2018
cut_v = 1031 #3330
X_train_1 = X_train + X_test[:cut_v]
y_train_1 = y_train + y_test[:cut_v]

X_test_1= X_test[cut_v:]
y_test_1 = y_test[cut_v:]

print(len(X_train_1), len(y_train_1), len(X_test_1), len(y_test_1))


# In[12]:


X_train_new = []
y_train_new = []

for i in range(len(X_train_1)):
    if y_train_1[i] >= -0.3 and y_train_1[i] <= 0.22:
        X_train_new.append(X_train_1[i])
        y_train_new.append(y_train_1[i])

        
X_test_new = []
y_test_new = []

for i in range(len(X_test_1)):
    if y_test_1[i] >= -0.3 and y_test_1[i] <= 0.22:
        X_test_new.append(X_test_1[i])
        y_test_new.append(y_test_1[i])

        
print(len(X_train_new), len(y_train_new), len(X_test_new), len(y_test_new))
print(max(y_train_new), min(y_train_new), max(y_test_new), min(y_test_new))


# In[ ]:





# In[5]:



def get_emb_p(path):
    #w_emb = {}
    voc = []
    emb = []
    
    with open(path, 'r', encoding = 'utf-8') as f:
        d = f.readlines()
        
    for i in tqdm(range(1, len(d)), total = len(d)-1):
        char = d[i].split()[0]
        _emb = np.array([float(k) for k in d[i].split()[1:]])
        voc.append(char)
        emb.append(_emb)
        
    return np.array(voc), np.array(emb)

path = '/ids/datasets/chinesenlp/textfast/cc.zh.300.vec'
voc, emb = get_emb_p(path)


# In[9]:


def token(voc):
    word_to_idx = {w:i for i, w in enumerate(voc)}
    #word_to_idx['UNK'] = len(voc)
    #word_to_idx['PAD'] = len(voc) + 1
    idx_to_word = {i:w for w, i in word_to_idx.items()}
    
    return word_to_idx, idx_to_word

# In[24]:
def split_sentence(txt):
    sents = [k for k in J.cut(txt)]
    return sents


def token_sen(sentence, sentence_size, word_to_idx):
    #words = sentence.split()[:sentence_size]
    words = split_sentence(sentence)
    words = words[:sentence_size]
    words = [k.lower() for k in words]

    if len(words) < sentence_size:
        words.extend(['PAD']*(sentence_size-len(words)))

    sen_idx = []
    for word in words:
        if word in word_to_idx:
            sen_idx.append(word_to_idx[word])
        else:
            sen_idx.append(word_to_idx['UNK'])

    return np.array(sen_idx)



def token_doc(docs, doc_size, sen_size, word_to_idx):
    doc_idx_batch = []
    
    for doc in docs:
        doc_idx = []
        for sen in doc:
            sen_idx = token_sen(sen, sen_size, word_to_idx)
            doc_idx.append([k for k in sen_idx])
           
        doc_idx = doc_idx[:doc_size]
        
        if len(doc_idx) < doc_size:
            doc_pad = [[word_to_idx['PAD']] * sen_size] * (doc_size-len(doc_idx))
            doc_idx.extend(doc_pad)
            
        doc_idx_batch.append(np.array(doc_idx))
        
    doc_idx_batch = np.stack(doc_idx_batch, axis=0)
        
    return doc_idx_batch


# In[10]:


word_to_idx, idx_to_word = token(voc)


# In[ ]:





# In[14]:


def get_batch_seq_1(X_batch):
    
    doc_seq_len = []
    sen_seq_len = []
    for doc in X_batch:
        if 68893 in doc.T[0]:
            cur_doc_seq = list(doc.T[0]).index(68893)
        else:
            cur_doc_seq = len(doc)
        doc_seq_len.append(cur_doc_seq)
        
        for sen in doc:
            if 68893 in sen:
                cur_sen_seq = list(sen).index(68893)
            else:
                cur_sen_seq = len(sen)
            #if cur_sen_seq == 0:
                #cur_sen_seq = 1
            sen_seq_len.append(cur_sen_seq)
    
    doc_seq_len = np.array(doc_seq_len)
    sen_seq_len = np.array(sen_seq_len)
    return doc_seq_len, sen_seq_len


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## rnn layers

# In[3]:


import tensorflow as tf
import numpy as np


# In[5]:


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


# In[4]:



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





# In[ ]:





# In[ ]:





# ## atten layers

# In[4]:


import tensorflow as tf
import numpy as np


def atten_self(outputs, final_states, n_layer = None, scope = None):
    # Self dot Attention: (output.T * softmat(output*state)) * out
    
    # if outputs are bi-rnn
    with tf.variable_scope(scope or 'attention'):
        if type(outputs) == tuple:
            #out_w = tf.Variable(tf.random_normal([self.n_hidden * 2, self.n_class]))
            output = tf.concat([outputs[0], outputs[1]], 2)  # [None, len_seq, n_hidden*2]

            # concat h state
            if n_layer is not None:
                final_hidden_state = tf.concat([final_states[0][-1][1], final_states[1][-1][1]], 1) # [None, n_hidden*2]
            else:
                final_hidden_state = tf.concat([final_states[0][1], final_states[1][1]], 1) # [None, n_hidden*2]

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


# In[8]:


def atten_layer_1(outputs, atten_size, n_layer = None, l2reg = 0, seq_len = None, use_mask = False, scope = None):
    # atten_layer: 
    # (1) output_projection = fully_connect(output, atten_size, tanh)
    # (2) output.T * softmat(output_projection*atten_vector)
    # tanh(fully_connect(outputs to atten_size))
    # outputs: [None, seq, hidden]
    # atten_vect: [atten_size, ]

    with tf.variable_scope(scope or 'attention') as scope:
        if type(outputs) == tuple:
            output = tf.concat([outputs[0], outputs[1]], 2)  # [None, len_seq, n_hidden*2]
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
            soft_attn_weights = tf.squeeze(soft_attn_weights, 2)
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


# In[9]:


def atten_layer_2(outputs, atten_size, n_layer = None, l2reg = 0, seq_len = None, use_mask = False, scope = None):
    # atten_layer: 
    # (1) output_projection = tanh(output*w + b) * u
    # (2) output.T * softmat(output_projection)
    # outputs: [None, seq, hidden]
    # atten_vect: [atten_size, ]
    with tf.variable_scope(scope or 'attention') as scope:
        if type(outputs) == tuple:
            output = tf.concat([outputs[0], outputs[1]], 2)  # [None, len_seq, n_hidden*2]
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
    


# In[10]:


def atten_wrap():
     pass
    


# In[11]:


def attention():

    pass


# In[ ]:





# In[ ]:



# In[ ]:





# In[ ]:





# ## stack layers

# In[6]:


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



def stack_attn(x_emb, sen_attn_w, doc_attn_w, scope = None):
    
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





# ## model layers

# In[8]:


import tensorflow as tf
import numpy as np


# In[ ]:


# way 1 to get logits


# In[ ]:


#logits = model.get_logits() 
#    cost, optimizer = model.compute_cost()
#    accuracy = model.classify_metric()
#    saver, init = model.initializer()
#

# In[9]:


def clf_train_op(final_outputs, labels, ac_fn = None, lr = 0.001, l2reg = 0, n_class = 2):
    logits = tf.layers.dense(final_outputs, n_class, activation = ac_fn,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    
    Y_proba = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return optimizer, logits, cost, accuracy, Y_proba


# In[5]:


def reg_train_op(final_outputs, labels, is_train, ac_fn = None, lr = 0.01, l2reg = 0, n_output = 2):
    
    #outputs_1 = tf.layers.dense(final_outputs, 64, activation = ac_fn,
                             #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    
    #outputs_2 = tf.layers.dense(outputs_1, 8, activation = ac_fn,
                             #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    
    #outputs_1_drop = tf.layers.dropout(outputs_1, rate = 1-0.5, training = is_train) 
    logits = tf.layers.dense(final_outputs, n_output, activation = None,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2reg))
    
    logits_c = tf.clip_by_value(logits, -1.0, 1.0)
    labels_c = tf.clip_by_value(labels, -1.0, 1.0)
    base_cost = tf.reduce_mean(tf.square(logits_c - labels_c))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) ############get it
    cost = tf.add_n([base_cost] + reg_losses) 
        
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    
    return optimizer, logits, cost


# In[11]:


def clf_predict_op(X_pred, embed, Y_proba):
    Y_proba_val = Y_proba.eval(feed_dict={X: X_pred, embedding: embed})
    y_pred = np.argmax(Y_proba_val, axis=1)
    
    return y_pred


# In[12]:


def reg_predict_op(logits):
    
    logits_c = tf.clip_by_value(logits, -1.0, 1.0)
    logits_c = tf.squeeze(logits_c, -1)
    return logits_c


# In[13]:


def initializer():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() #
    
    return init, saver


# In[ ]:



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





# In[ ]:





# In[ ]:





# ## training section

# In[9]:


def get_b_t (X, y, b =1):
    
    idx = np.random.choice(len(X), b)
    
    X_b = [X[k] for k in idx]
    y_b = [y[k] for k in idx]
    
    return X_b, y_b


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


tf.reset_default_graph()
doc_size = 200
sen_size = 60
voc_size = 2000000
emb_size = 300
train_embed = 0
dense_n = 300


with tf.name_scope('inputs'):

    X = tf.placeholder(shape=(None, doc_size, sen_size), dtype=tf.int64, name='inputs') #len_seq = 40
    y = tf.placeholder(shape=(None,), dtype=tf.float32, name='labels')
    is_training = tf.placeholder_with_default(False, shape = [], name='is_training')
    doc_seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='doc_seq_length')
    sen_seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='sen_seq_length')
    #X_embed = tf.placeholder(shape=(None, doc_size, sen_size, emb_size), dtype=tf.float32, name='x_emb')
    rank_pair = tf.placeholder(tf.float32, shape=(None, None), name = 'rank_pair')
    emb_holder = tf.placeholder(tf.float32, shape = [voc_size, emb_size])
        
with tf.device('/cpu:0'):
    with tf.name_scope('embedding'):
        if train_embed == 1:
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
                                    trainable = True)
        else:
            #embedding = tf.get_variable('embedding', [vocab_size, embedding_size], trainable=False)
            embedding = tf.get_variable('embedding', [voc_size, emb_size], trainable = True) # initializer = emb

        X_embed = tf.nn.embedding_lookup(embedding, X) # None, doc_s, sen_s, embed_s

        assign_emb = embedding.assign(emb_holder) 
        
        
with tf.device('/gpu:0'):

    with tf.name_scope('sen_rnn'):
        X_embed_reshape = tf.reshape(X_embed, [-1, sen_size, emb_size])
        sen_rnn_outputs, sen_rnn_states = bi_rnn(X_embed_reshape, n_hidden = 50, seq_len = sen_seq_length, n_layer = 3, 
                                         is_train = is_training, 
                                         keep_prob = 0.5, scope = 'sen_rnn_block')

    with tf.name_scope('sen_attn'):
        sen_atten_out, sen_atten_w = atten_layer_1(sen_rnn_outputs, atten_size = 50, n_layer = 3, l2reg = 0.001, 
                                                      seq_len = sen_seq_length, use_mask = True, scope = 'sen_attn_block')
        # Dropout
        #sen_atten_out_drop = tf.layers.dropout(sen_atten_out, rate = 1-0.5, training = is_training) # tf.nn.dropout


    with tf.name_scope('sen_stack'):
        sen_outs = stack_layer(sen_atten_out, sen_atten_w, sen_rnn_states, X_embed_reshape, scope = 'sen_stack_block')
        #sen_outs = stack_layer(sen_atten_out, sen_rnn_states, scope = 'sen_stack_block')
        #sen_outs_drop = tf.layers.dropout(sen_outs, rate = 1-0.5, training = is_training)

with tf.device('/gpu:1'):
    with tf.name_scope('doc_rnn'):
        doc_inputs = tf.reshape(sen_outs, [-1, doc_size, sen_outs.shape[1]])
        #doc_inputs = tf.reshape(sen_outs, [-1, doc_size, sen_outs.shape[1]])
        doc_rnn_outputs, doc_rnn_states = bi_rnn(doc_inputs, n_hidden = 50, seq_len = doc_seq_length, n_layer = 3, 
                                         is_train = is_training, 
                                         keep_prob = 0.5, scope = 'doc_rnn_block')

    with tf.name_scope('doc_attn'):
        doc_atten_out, doc_atten_w = atten_layer_1(doc_rnn_outputs, atten_size = 50, n_layer = 3, l2reg = 0.001, 
                                                      seq_len = doc_seq_length, use_mask = True, scope = 'doc_attn_block')
        # Dropout
        #doc_atten_out_drop = tf.layers.dropout(doc_atten_out, rate = 1-0.5, training = is_training) # tf.nn.dropout


    with tf.name_scope('doc_stack'):
        doc_outs = stack_layer(doc_atten_out, doc_atten_w, doc_rnn_states, doc_inputs, scope = 'doc_stack_block')
        #doc_outs = stack_layer(doc_atten_out, doc_rnn_states, scope = 'doc_stack_block')
        #doc_outs_drop = tf.layers.dropout(doc_outs, rate = 1-0.5, training = is_training)

    with tf.name_scope('attn_stack'):    
        doc_emb = stack_attn(X_embed_reshape, sen_atten_w, doc_atten_w, scope = None) # [None, doc]


with tf.device('/gpu:2'):
    with tf.name_scope('final_out'):
        #doc_emb_dense = tf.layers.dense(doc_emb, dense_n, activation = None)
        final_outputs = tf.concat([doc_outs, doc_emb], axis = -1)
        #final_outputs = doc_emb
        final_outputs_dense = tf.layers.dense(final_outputs, dense_n, activation = tf.nn.relu)
        final_outputs_drop = tf.layers.dropout(final_outputs_dense, rate = 1-0.5, training = is_training)

    with tf.name_scope('logits'):
        optimizer, logits, cost = reg_train_op(final_outputs_drop, y, is_train = is_training, 
                                               ac_fn = tf.nn.tanh, lr = 1e-4, l2reg = 0.001, n_output = 1)
        rank_cor = my_corr(rank_pair)
        init, saver = initializer()

    with tf.name_scope('prediction'):
        y_pred = reg_predict_op(logits)


# In[13]:


#print(X_embed, '\n', X_embed_reshape)
#print('\n\n')
#print(sen_atten_out, '\n', sen_atten_w, '\n')
#print('\n\n')
#print(doc_inputs, '\n')
#print(doc_atten_out, '\n', doc_atten_w, '\n')
#print('\n\n')
#print(doc_emb, '\n', final_outputs, '\n', final_outputs_dense)
#print('\n\n')
#print(logits,'\n',  cost)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


print('......Start training......')


# In[22]:


epochs = 30


# In[ ]:


loss_record = {}
acc_record = {}
rank_record = {}
batch_size = 32
#with tf.Session(config=session_conf) as sess:
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    #sess.run(init)
    #sess.run(embedding.assign(emb_holder), {emb_holder: emb})

    saver.restore(sess, '/ids/datasets/chinesenlp/textfast/saved_model/best_model_v2')
    min_loss = np.inf

    for epoch in range(epochs + 1):
        loss_train = 0
        loss_test = 0
        acc_train = 0
        acc_test = 0
        rank_train = 0
        rank_test = 0

        #print("epoch: {}\t".format(epoch), end="")

        # training
        n_batch = len(X_train_new)//batch_size
        for _ in tqdm(range(n_batch), total = n_batch):
            X_batch, y_batch = get_b_t(X_train_new, y_train_new, batch_size)
            X_batch_tok = token_doc(X_batch, doc_size, sen_size, word_to_idx)
            doc_seq_len, sen_seq_len = get_batch_seq_1(X_batch_tok)
            #X_batch_emb, doc_seq_len, sen_seq_len = get_batch_emb(X_batch, doc_size, sen_size)
            _, loss, sen_alpha, doc_alpha, y_pred_val = sess.run([optimizer, cost, sen_atten_w, doc_atten_w, y_pred],
                                          feed_dict={X: X_batch_tok, y: y_batch, 
                                                     doc_seq_length: doc_seq_len, 
                                                     sen_seq_length: sen_seq_len, is_training: True}) 
            #rank_val = sess.run(my_corr, feed_dict = {rank_pair: [y_pred_val, y_batch]})
            rank_val = spearmanr(np.squeeze(y_pred_val), y_batch)[0]
            rank_train += rank_val
            #acc_train += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_train += loss
        #acc_train /= n_batch
        rank_train /= n_batch
        loss_train /= n_batch

        #if (epoch + 1) % (epochs//10) == 0:
        #if (epoch + 0) % 10 == 0:
            #print('Epoch:', '%d' % (epoch + 0), 'cost =', '{:.6f}'.format(loss))

        # testing
        y_pred_vals = []
        n_batch = len(X_test_new)//batch_size
        for i in tqdm(range(n_batch), total = n_batch):
            if i == n_batch - 1:
                X_batch = X_test_new[i*batch_size:]
                y_batch = y_test_new[i*batch_size:]
            else:
                X_batch = X_test_new[i*batch_size:(i+1)*batch_size]
                y_batch = y_test_new[i*batch_size:(i+1)*batch_size]
            #X_batch, y_batch = get_b_t(X_test_new, y_test_new, batch_size)
            X_batch_tok = token_doc(X_batch, doc_size, sen_size, word_to_idx)
            doc_seq_len, sen_seq_len = get_batch_seq_1(X_batch_tok)
            loss, sen_alpha, doc_alpha, y_pred_val = sess.run([cost, sen_atten_w, doc_atten_w, y_pred],
                                          feed_dict={X: X_batch_tok, y: y_batch, 
                                                     doc_seq_length: doc_seq_len, 
                                                     sen_seq_length: sen_seq_len,
                                                     is_training: False})
            #rank_val = sess.run(my_corr, feed_dict = {rank_pair: [y_pred_val, y_batch]})
            y_pred_vals.extend(list(np.squeeze(y_pred_val)))
            #rank_val = spearmanr(np.squeeze(y_pred_val), y_batch)[0]
            #rank_test += rank_val
            #acc_test += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_test += loss
        #acc_test /= n_batch
        #rank_test /= n_batch
        loss_test /= n_batch
        rank_test = spearmanr(y_pred_vals, y_test_new)[0]
        

        loss_record[epoch] = [loss_train, loss_test]
        acc_record[epoch] = [acc_train, acc_test]
        rank_record[epoch] = [rank_train, rank_test]

        if epoch % 1 == 0:
            print("epoch: {}\t".format(epoch), end="")
            print("loss: {:.5f}, val_loss: {:.5f}, train_rank: {:.5f}, test_rank: {:.5f}".format(
                loss_train, loss_test, rank_train, rank_test))


        if loss_test < min_loss:
            saver.save(sess, '/ids/datasets/chinesenlp/textfast/saved_model/best_model_v2')
            min_loss = loss_test

    saver.save(sess, '/ids/datasets/chinesenlp/textfast/saved_model/final_model_v2')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




