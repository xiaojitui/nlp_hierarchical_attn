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

from embedding import embedding_bert
from model import utils, model, rnn_layer, attn_layer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore Info and Warning
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

# In[ ]:





# In[ ]:





# ## load data

# In[2]:


path = './samples/en/doc_level'
#os.listdir(path)


# In[3]:


with open(os.path.join(path, 'pos.pkl'), 'rb') as f:
    pos_data = pickle.load(f)
with open(os.path.join(path, 'neg.pkl'), 'rb') as f:
    neg_data = pickle.load(f)
len(pos_data), len(neg_data)


# In[ ]:





# In[ ]:





# ## preprocess

# In[4]:


with open('./model/contractions.pkl', 'rb') as f:
    contractions = pickle.load(f)


# In[5]:


# clean, contract, split to sentence, token - if in dict, ok; if not in dict, replace with <digit>


# In[6]:


pos_data_clean = []

for x in tqdm(pos_data, total = len(pos_data)):
    x = utils.clean_sentence([x])
    x = utils.replace_contr(x, contractions)
    #x = utils.split_document(x[0])
    #x, _ = utils.refine_document(x) # use this if need to train "phrase"
    pos_data_clean.append(x[0])
    
neg_data_clean = []

for x in tqdm(neg_data, total = len(neg_data)):
    x = utils.clean_sentence([x])
    x = utils.replace_contr(x, contractions)
    #x = utils.split_document(x[0])
    #x, _ = utils.refine_document(x) # use this if need to train "phrase"
    neg_data_clean.append(x[0])
    
#len(pos_data_clean), len(neg_data_clean)


# In[ ]:





# In[7]:


train_size = int((12500+12500)*0.85)
test_size = int((12500+12500)*0.15)
#train_size, test_size


# In[ ]:





# In[8]:



pos_train_idx = np.random.choice(12500, train_size//2, replace=False)
pos_train = [pos_data_clean[k] for k in pos_train_idx]
pos_test = [pos_data_clean[k] for k in range(12500) if k not in pos_train_idx]


neg_train_idx = np.random.choice(12500, train_size//2, replace=False)
neg_train = [neg_data_clean[k] for k in neg_train_idx]
neg_test = [neg_data_clean[k] for k in range(12500) if k not in neg_train_idx]

X_train = pos_train + neg_train
y_train = [1]*len(pos_train) + [0]*len(neg_train)
X_test = pos_test + neg_test
y_test = [1]*len(pos_test) + [0]*len(neg_test)


idx = np.arange(len(X_train))
np.random.shuffle(idx)
X_train = [X_train[k] for k in idx]
y_train = [y_train[k] for k in idx]


idx = np.arange(len(X_test))
np.random.shuffle(idx)
X_test = [X_test[k] for k in idx]
y_test = [y_test[k] for k in idx]

print(len(X_train), len(X_test), len(y_train), len(y_test))


# In[ ]:





# In[ ]:





# ## prepare doc, sen, and study sample statistics

# In[9]:


# convert data to sentences, then phrases
X_train_doc = []
for x in tqdm(X_train, total = len(X_train)):
    doc = utils.split_document(x)
    phrase, _ = utils.refine_document(doc) 
    phrase_g = []
    [phrase_g.extend(ele) for ele in phrase]
    X_train_doc.append(phrase_g)

    
# refine , if necessary
#X_train_doc, _ = utils.refine_documemt(X_train_doc, contractions)


X_test_doc = []
for x in tqdm(X_test, total = len(X_test)):
    doc = utils.split_document(x)
    phrase, _ = utils.refine_document(doc) 
    phrase_g = []
    [phrase_g.extend(ele) for ele in phrase]
    X_test_doc.append(phrase_g)

    
#X_test_doc, _ = utils.refine_documemt(X_test_doc, contractions)

# refine , if necessary
print(len(X_train), len(X_train_doc), len(X_test), len(X_test_doc))


# In[10]:


#idx = np.random.choice(21250)
#X_train_doc[idx], X_train[idx], y_train[idx]


# In[11]:


#idx = np.random.choice(3750)
#X_test_doc[idx], X_test[idx], y_test[idx]


# In[12]:


def get_phrase_stat(docs, doc_cut_thres = 0.90, sen_cut_thres = 0.90):
    doc_len = []
    sen_len = []
    
    for doc in docs:
        doc_len.append(len(doc))
        
        for sen in doc:
            sen_len.append(len(sen))
            
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


# In[ ]:





# In[13]:


#doc_cut_val, sen_cut_val, a, b = utils.get_doc_stat(X_train_doc + X_test_doc, doc_cut_thres = 0.95, sen_cut_thres = 0.95)
#doc_cut_val, sen_cut_val


# In[ ]:





# In[14]:


#doc_cut_val, sen_cut_val, _, _ = utils.get_doc_stat(X_train_doc + X_test_doc, doc_cut_thres = 0.90, sen_cut_thres = 0.90)
#doc_cut_val, sen_cut_val


# In[15]:


# choose doc_len = 70, sen_len = 25


# In[ ]:





# ## prepare voc, token, emb

# In[16]:


bert_path = './bert/checkpoint/en'
bert_config_file = './bert/checkpoint/en/bert_config.json'
bert_vocab_file = './bert/checkpoint/en/vocab.txt'
init_checkpoint = './bert/checkpoint/en/bert_model.ckpt'
graph_file = './bert/tmp_en/graph_doc_dt' # where to put temporary graph
model_dir = './bert/tmp_1' # where to put temporary graph
select_layers = [-4, -3, -2, -1] 
#select_layers = [0] # get the cls token
doc_len = 70
sen_len = 25

batch_size = 32


# In[18]:


tokenizer, estimator = embedding_bert.prepare_bert(bert_vocab_file, bert_config_file, init_checkpoint, sen_len, 
                                                   select_layers, batch_size, graph_file, model_dir)


# In[19]:






embedding_size = 3072
doc_len = 70
sen_len = 25
keep_prob = 0.5

epochs = 10 #20
batch_size = 32 #64

n_hidden = 50
n_layer = 2

atten_size = 50
l2reg = 0.0 #1e-5
use_mask = True
sen_CLS = False

n_class = 2
lr = 1e-4

savepath = '/ids/datasets/CNN_datasets/saved_weights_n/best_model.ckpt'
finalpath = '/ids/datasets/CNN_datasets/saved_weights_n/final_model.ckpt'


# In[29]:


# generate doc emb
def get_doc_emb(x_emb, doc_attn_w, scope = None):
    
    with tf.variable_scope(scope or 'stack_attn') as scope:
        # X_emb = [None, doc, embedding]
        # doc_attn_w = [None, doc]
        doc_attn_w = tf.expand_dims(doc_attn_w, -1) #[None, doc, 1]
        doc_emb = tf.reduce_sum(doc_attn_w * X_emb, axis = 1) # [None, embedding]
        
    return doc_emb


# In[30]:


tf.reset_default_graph()

with tf.name_scope('inputs'):
    
    ##X = tf.placeholder(shape=(None, sen_len), dtype=tf.int64, name='inputs')
    X_emb = tf.placeholder(shape=(None, doc_len, 3072), dtype=tf.float32, name='inputs')
    y = tf.placeholder(shape=(None,), dtype=tf.int64, name='labels')
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
    optimizer, logits, cost, accuracy, Y_proba = model.clf_train_op(doc_out_drop, y, ac_fn = None, 
                                                                    lr = lr, l2reg = l2reg, n_class = n_class) 
    init, saver = model.initializer()
    


# In[17]:





# In[32]:


print(X_emb)
print('\n\n')
print(atten_out, '\n', soft_atten_weights)
print('\n\n')
print(doc_emb, '\n', doc_out, '\n', doc_out_1)
print('\n\n')
print(logits,'\n',  cost)


# In[ ]:





# In[ ]:





# In[ ]:


loss_record = {}
acc_record = {}
#with tf.Session(config=session_conf) as sess:
with tf.Session() as sess:
    sess.run(init)
    
    #saver.restore(sess, finalpath)
    min_loss = np.inf

    for epoch in range(epochs):
        t0 = time.time()
        
        loss_train = 0
        loss_test = 0
        acc_train = 0
        acc_test = 0

        #print("epoch: {}\t".format(epoch), end="")

        # training
        n_batch = len(X_train)//batch_size
        #for _ in range(n_batch):
        for _ in tqdm(range(n_batch), total = n_batch):
            X_batch, y_batch = utils.get_batch_data(X_train_doc, y_train, batch_size)
            X_batch_emb = embedding_bert.get_batch_emb(X_batch, doc_len, sen_len, tokenizer, estimator)
            X_batch_emb = X_batch_emb[:, :, 0, :]
            X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            _, loss, acc, atten_w = sess.run([optimizer, cost, accuracy, soft_atten_weights],
                                          feed_dict={X_emb: X_batch_emb, y: y_batch, seq_length: X_batch_seq, is_training:True}) #seq_length: batch_seq_len

            acc_train += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_train += loss
        acc_train /= n_batch

        #if (epoch + 1) % (epochs//10) == 0:
        #if (epoch + 0) % 10 == 0:
            #print('Epoch:', '%d' % (epoch + 0), 'cost =', '{:.6f}'.format(loss))

        # testing
        n_batch = len(X_test)//batch_size
        #for i in range(n_batch):
        for i in tqdm(range(n_batch), total = n_batch):
            X_batch, y_batch = utils.get_batch_test(X_test_doc, y_test, i, batch_size)
            X_batch_emb = embedding_bert.get_batch_emb(X_batch, doc_len, sen_len, tokenizer, estimator)
            X_batch_emb = X_batch_emb[:, :, 0, :]
            X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            loss, acc = sess.run([cost, accuracy],
                                          feed_dict={X_emb: X_batch_emb, y: y_batch, seq_length: X_batch_seq, is_training:False}) #seq_length: batch_seq_len
            acc_test += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_test += loss
        acc_test /= n_batch

        loss_record[epoch] = [loss_train, loss_test]
        acc_record[epoch] = [acc_train, acc_test]

        #if epoch ==0 or (epoch + 1) % 4 == 0:
        if epoch >= 0:
            print("epoch: {}\t".format(epoch), end="")
            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, acc_train, acc_test), end = "\t")
            print(round(time.time() - t0, 3))
            


        if loss_test < min_loss:
            saver.save(sess, savepath)
            min_loss = loss_test

    saver.save(sess,  finalpath)


# In[ ]:




