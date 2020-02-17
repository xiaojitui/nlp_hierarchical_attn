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


# In[ ]:





# In[2]:


## need to update X_test_batch function


# In[ ]:





# ## load data

# In[3]:


path = './samples/en/sen_level'
os.listdir(path)


# In[4]:


train_data = pd.read_csv(os.path.join(path, 'train_binary_sent.csv'))
dev_data = pd.read_csv(os.path.join(path, 'dev_binary_sent.csv'))
test_data = pd.read_csv(os.path.join(path, 'test_binary_sent.csv'))

train_data.shape, test_data.shape, dev_data.shape


# In[5]:


train_data.head()


# In[6]:


train_data.label.value_counts()


# In[7]:


for i in range(20):
    print(i, '\t', train_data.iloc[i, 1])


# In[8]:


X_train = train_data['sentence'].values
y_train = train_data['label'].values
X_test = test_data['sentence'].values
y_test = test_data['label'].values

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## study sample statistics

# In[9]:


# X_train is 1D list = [sent1, sen2, sen3.....]
cut_val, sen_len = utils.get_sen_stat(X_train, cut_thres = 0.95)
cut_val


# In[10]:


# choose sen_len = 30


# In[ ]:





# In[ ]:





# ## prepare voc, token, emb

# In[11]:


with open('./model/contractions.pkl', 'rb') as f:
    contractions = pickle.load(f)


# In[ ]:





# ### emb

# In[12]:


import time
t0 = time.time()
path = '/ids/datasets/glove_vectors/glove.6B.300d.txt' # can use 300d
word_to_idx, words, emb = utils.load_embedding(path)
print(time.time() - t0)
emb.shape, len(word_to_idx), len(words)
# last two: UNK, PAD


# In[ ]:





# ### test section: optional

# In[ ]:





# In[11]:


xx = [train_data.iloc[0, 1], train_data.iloc[20, 1], train_data.iloc[27, 1], train_data.iloc[33, 1]]
xx


# In[12]:


# input must be 2D list
xxx = utils.prepare_words([xx], contractions)
np.array(xxx)


# In[ ]:





# In[13]:


xx_clean = utils.replace_contr(utils.clean_sentence(xx), contractions)
xx_clean


# In[14]:


xx_clean = utils.replace_contr(xx, contractions)
xx_clean


# In[15]:


x = utils.token_sens(xx_clean, sentence_size = 30, word_to_idx = word_to_idx)
x.shape


# In[ ]:





# In[16]:


a, b = utils.get_batch_data(X_train, y_train, batch_size = 5)
a_clean = utils.replace_contr(a, contractions)
a_clean.shape, b.shape


# In[17]:


a, a_clean, b


# In[ ]:





# In[18]:


a_emb = utils.token_sens(a_clean, sentence_size = 10, word_to_idx = word_to_idx)
c = utils.get_sen_batch_seq(a_emb,  pad_token = 400001)
c


# In[19]:


a_emb


# In[ ]:





# In[ ]:


workflow: batch, token, seq, emb == train


# In[ ]:


a, b = utils.get_batch_data(X_train, y_train, batch_size = 5)
a_clean = utils.replace_contr(a, contractions)
a_emb = utils.token_sens(a_clean, sentence_size = 10, word_to_idx = word_to_idx)
c = utils.get_sen_batch_seq(a_emb,  pad_token = 400001)


# In[ ]:





# ## Train section

# In[13]:


vocab_size = emb.shape[0]
embedding_size = emb.shape[1]
pretrained_emb = True
finetune_emb = False
# doc_len = 1
sen_len = 30
keep_prob = 0.5

epochs = 50 #20
batch_size = 64

n_hidden = 64
n_layer = 3

atten_size = 64
l2reg = 1e-5
use_mask = True

n_class = 2
lr = 1e-4

savepath = './saved_weights/best_model.ckpt'
finalpath = './saved_weights/final_model.ckpt'


# In[14]:


tf.reset_default_graph()

with tf.name_scope('inputs'):
    
    X = tf.placeholder(shape=(None, sen_len), dtype=tf.int64, name='inputs')
    y = tf.placeholder(shape=(None,), dtype=tf.int64, name='labels')
    is_training = tf.placeholder_with_default(False, shape = [], name='is_training')
    seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='seq_length')
    
## prepare embedding
#with tf.device('/cpu:0'):
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


with tf.name_scope('rnn_layer'):
    rnn_outputs, rnn_states = rnn_layer.bi_rnn(X_embed, n_hidden = n_hidden, seq_len = seq_length, n_layer = n_layer, is_train = is_training, 
                                               keep_prob = keep_prob) #### need seq_length??
    
with tf.name_scope('attention_layer'):
    atten_out, soft_atten_weights = attn_layer.atten_layer_project(rnn_outputs, atten_size, n_layer = n_layer, 
                                                                   l2reg = l2reg, seq_len = seq_length, use_mask = use_mask)
    # Dropout
    atten_out_drop = tf.nn.dropout(atten_out, keep_prob)
    
    
with tf.name_scope('logits'):
    optimizer, logits, cost, accuracy, Y_proba = model.clf_train_op(atten_out_drop, y, ac_fn = tf.nn.relu, 
                                                                    lr = lr, l2reg = l2reg, n_class = n_class) 
    init, saver = model.initializer()
    


# In[15]:


print(X_embed)
print('\n\n')
print(atten_out, '\n', soft_atten_weights)
print('\n\n')
print(logits,'\n',  cost)


# In[ ]:





# In[16]:


loss_record = {}
acc_record = {}
#with tf.Session(config=session_conf) as sess:
with tf.Session() as sess:
    sess.run(init)
    
    #saver.restore(sess, finalpath)
    min_loss = np.inf

    for epoch in range(epochs):
        loss_train = 0
        loss_test = 0
        acc_train = 0
        acc_test = 0

        #print("epoch: {}\t".format(epoch), end="")

        # training
        n_batch = X_train.shape[0]//batch_size
        for _ in range(n_batch):
            X_batch, y_batch = utils.get_batch_data(X_train, y_train, batch_size)
            X_batch = utils.replace_contr(X_batch, contractions)
            X_batch_emb = utils.token_sens(X_batch, sen_len, word_to_idx)
            X_batch_seq = utils.get_sen_batch_seq(X_batch_emb,  pad_token = 400001)
            #batch_seq_len = np.array([list(x).index(0) + 1 for x in X_batch])  # actual lengths of sequences
            _, loss, acc, atten_w = sess.run([optimizer, cost, accuracy, soft_atten_weights],
                                          feed_dict={X: X_batch_emb, y: y_batch, seq_length: X_batch_seq, is_training:True}) #seq_length: batch_seq_len

            acc_train += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_train += loss
        acc_train /= n_batch

        #if (epoch + 1) % (epochs//10) == 0:
        #if (epoch + 0) % 10 == 0:
            #print('Epoch:', '%d' % (epoch + 0), 'cost =', '{:.6f}'.format(loss))

        # testing
        n_batch = X_test.shape[0]//batch_size
        for i in range(n_batch):
            X_batch, y_batch = utils.get_batch_test(X_test, y_test, i, batch_size)
            
            X_batch = utils.replace_contr(X_batch, contractions)
            X_batch_emb = utils.token_sens(X_batch, sen_len, word_to_idx)
            X_batch_seq = utils.get_sen_batch_seq(X_batch_emb,  pad_token = 400001)
            #batch_seq_len = np.array([list(x).index(0) + 1 for x in X_batch])
            loss, acc, atten_w = sess.run([cost, accuracy, soft_atten_weights],
                                          feed_dict={X: X_batch_emb, y: y_batch, seq_length: X_batch_seq, is_training:False}) #seq_length: batch_seq_len
            acc_test += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_test += loss
        acc_test /= n_batch

        loss_record[epoch] = [loss_train, loss_test]
        acc_record[epoch] = [acc_train, acc_test]

        if epoch ==0 or (epoch + 1) % 10 == 0:
        #if epoch >= 0:
            print("epoch: {}\t".format(epoch), end="")
            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, acc_train, acc_test))


        if loss_test < min_loss:
            saver.save(sess, savepath)
            min_loss = loss_test

    saver.save(sess,  finalpath)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Predict section

# In[ ]:


with tf.Session() as sess:
    saver.restore(sess, savepath)
    
    X_batch, y_batch = utils.get_batch_data(X_test, y_test, 10)
    X_batch = utils.replace_contr(X_batch, contractions)
    X_batch_emb = utils.token_sens(X_batch, sen_len, word_to_idx)
    X_batch_seq = utils.get_sen_batch_seq(X_batch_emb,  pad_token = 400001)
    #batch_seq_len = np.array([list(x).index(0) + 1 for x in X_batch])
    atten_w, y_proba = sess.run([soft_atten_weights, Y_proba], feed_dict={X: X_batch_emb, seq_length: X_batch_seq, is_training:False})
    
    y_pred = np.argmax(y_proba, axis = 1)


# In[99]:


X_batch, y_batch, y_pred


# In[100]:


X_batch_seq


# In[101]:


atten_w[0]


# In[104]:


idx = 5
s = utils.split_sentence(X_batch[idx])
for i in range(len(s)):
    print(s[i], round(atten_w[idx][i], 3), end = ';')
fig = plt.figure(figsize=(20, 5)) 
ax = fig.add_subplot(1, 1, 1)
cax = ax.matshow(atten_w[idx][:, np.newaxis].T, cmap='viridis')
fig.colorbar(cax, orientation = 'horizontal', shrink = 0.2)
ax.set_xticks(np.arange(sen_len))
ax.tick_params(length=0)
ax.set_xticklabels([k for k in s], fontdict={'fontsize': 12}, rotation=40, ha='left')
ax.set_yticklabels([''])

#ax.set_xticklabels([k for k in tests[0]], fontdict={'fontsize': 14}, rotation=90)
#ax.set_yticklabels([''] + ['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6'], fontdict={'fontsize': 14})
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## when OOM - tfsensor dimension too large
############ loading #############
voc, emb = load_voc_emb(path)
word_to_idx, idx_to_word = token(voc)


if emb.shape is too large, 
cannot do: embedding = tf.get_variable('embedding', initializer = emb, trainable = finetune_emb)
because initializer is too large
need to do: embedding = tf.get_variable('embedding', [voc_size, emb_size], trainable = finetune_emb) #only indicate the shape


############ in construction #############
with tf.name_scope('inputs'):
    # X =
    # y = 
    emb_holder = tf.placeholder(tf.float32, shape = [voc_size, emb_size])

with tf.name_scope('embedding'):
    if pretrained_emb is False:
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable = True)
    else:
        embedding = tf.get_variable('embedding', [vocab_size, embedding_size], trainable = finetune_emb) ############### this is the change
    X_embed = tf.nn.embedding_lookup(embedding, X) # None, doc_s, sen_s, embed_s


############# in execution ############### 
with tf.Session() as sess:
    sess.run(init)
    sess.run(embedding.assign(emb_holder), {emb_holder: emb})
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pickle
from tqdm import tqdm
#with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2013_2018.pkl', 'rb') as f:
#with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2010_2015.pkl', 'rb') as f:
    #train_data = pickle.load(f)
        
#with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2010_2012.pkl', 'rb') as f:
#with open('/ids/datasets/chinesenlp/wind/train/training_wind_full_2016_2018.pkl', 'rb') as f:
    #test_data = pickle.load(f)
    
with open('/ids/datasets/chinesenlp/wind/backup/x_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('/ids/datasets/chinesenlp/wind/backup/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
    
with open('/ids/datasets/chinesenlp/wind/backup/x_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
    
with open('/ids/datasets/chinesenlp/wind/backup/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)    


# In[ ]:





# In[ ]:





# In[ ]:


#vocab_size = config['vocab_size']
vocab_size = emb.shape[0]
#embedding_size = config['embedding_size']
embedding_size = emb.shape[1]
train_embed = config['train_embed']
#len_seq = config['len_seq']
len_seq = 30
keep_prob = config['keep_prob']
#epochs = config['epochs']
epochs = 10
#batch_size = config['batch_size']
batch_size = 32

n_hidden = config['n_hidden']
n_layer = config['n_layer']

atten_size = 50
l2reg = 0.001

n_class = 2

savepath = config['savepath']
finalpath = config['finalpath']


# In[ ]:


X_train_idx = token_batch(X_train, 30, word_to_idx)
X_test_idx = token_batch(X_test, 30, word_to_idx)
X_dev_idx = token_batch(X_dev, 30, word_to_idx)
X_train_idx.shape, X_test_idx.shape, X_dev_idx.shape


# In[ ]:


tf.reset_default_graph()

with tf.name_scope('inputs'):
    
    X = tf.placeholder(shape=(None, len_seq), dtype=tf.int64, name='inputs')
    y = tf.placeholder(shape=(None,), dtype=tf.int64, name='labels')
    is_training = tf.placeholder_with_default(False, shape = [], name='is_training')
    seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='seq_length')
    
## prepare embedding
#with tf.device('/cpu:0'):
with tf.name_scope('embedding'):
    if train_embed == 1:
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
                                trainable = True)
    else:
        #embedding = tf.get_variable('embedding', [vocab_size, embedding_size], trainable=False)
        embedding = tf.get_variable('embedding', initializer = emb, trainable=False)
    
    X_embed = tf.nn.embedding_lookup(embedding, X) # None, doc_s, sen_s, embed_s


with tf.name_scope('rnn_layer'):
    rnn_outputs, rnn_states = bi_rnn(X_embed, n_hidden = 64, n_layer = 2, keep_prob = 0.5)
    
with tf.name_scope('attention_layer'):
    atten_out, soft_atten_weights = atten_layer_1(rnn_outputs, atten_size, n_layer, l2reg)
    # Dropout
    atten_out_drop = tf.nn.dropout(atten_out, keep_prob)
    
    
with tf.name_scope('logits'):
    optimizer, logits, cost, accuracy, Y_proba = clf_train_op(atten_out_drop, y, ac_fn = tf.nn.relu, 
                                                                    lr = 0.001, l2reg = l2reg, n_class = 2) 
    init, saver = initializer()
    


# In[ ]:


loss_record = {}
acc_record = {}
#with tf.Session(config=session_conf) as sess:
with tf.Session() as sess:
    sess.run(init)

    min_loss = np.inf

    for epoch in range(epochs):
        loss_train = 0
        loss_test = 0
        acc_train = 0
        acc_test = 0

        #print("epoch: {}\t".format(epoch), end="")

        # training
        n_batch = X_train.shape[0]//batch_size
        for _ in range(n_batch):
            X_batch, y_batch = get_batch_data(X_train_idx, y_train, batch_size)
            #batch_seq_len = np.array([list(x).index(0) + 1 for x in X_batch])  # actual lengths of sequences
            _, loss, acc, atten_w = sess.run([optimizer, cost, accuracy, soft_atten_weights],
                                          feed_dict={X: X_batch, y: y_batch}) #seq_length: batch_seq_len

            acc_train += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_train += loss
        acc_train /= n_batch

        #if (epoch + 1) % (epochs//10) == 0:
        #if (epoch + 0) % 10 == 0:
            #print('Epoch:', '%d' % (epoch + 0), 'cost =', '{:.6f}'.format(loss))

        # testing
        n_batch = X_test.shape[0]//batch_size
        for _ in range(n_batch):
            X_batch, y_batch = get_batch_data(X_test_idx, y_test, batch_size)
            #batch_seq_len = np.array([list(x).index(0) + 1 for x in X_batch])
            loss, acc, atten_w = sess.run([cost, accuracy, soft_atten_weights],
                                          feed_dict={X: X_batch, y: y_batch}) #seq_length: batch_seq_len
            acc_test += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_test += loss
        acc_test /= n_batch

        loss_record[epoch] = [loss_train, loss_test]
        acc_record[epoch] = [acc_train, acc_test]

        if epoch ==0 or (epoch + 1) % 10 == 0:
            print("epoch: {}\t".format(epoch), end="")
            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, acc_train, acc_test))


        if loss_test < min_loss:
            saver.save(sess, './saved_model')
            min_loss = loss_test

    saver.save(sess,  './final_model')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## be section


# In[ ]:


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

len(X_train), len(y_train), len(X_test), len(y_test)


# In[ ]:


def get_b_t (X, y, b =1):
    
    idx = np.random.choice(len(X), b)
    
    X_b = [X[k] for k in idx]
    y_b = [y[k] for k in idx]
    
    return np.array(X_b), np.array(y_b)


# In[ ]:


X = [['我想去', '看电影', '中文自然语言处理', '包括字向量'], 
          ['包括字向量', '中文自然语言处理包括字向量'], 
          ['我想', '看电', '中文语言处理', '字向量', '语言处理', '中文语言', '中'], 
     ['我想去', '中文语言处理'], 
    ['看电影', '中处理']]
y = [-0.1, 0.22, 0.5, -0.3, 0.4]

X_train = X[:3]
y_train = y[:3]
X_test = X[3:]
y_test = y[3:]


# In[ ]:





# In[ ]:


#tf.reset_default_graph()
doc_size = 200
sen_size = 100
emb_size = 768*4
dense_n = 300


tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)
#g_1 = tf.Graph()

#with g_1.as_default():

with tf.name_scope('inputs'):

    X = tf.placeholder(shape=(None, doc_size, sen_size), dtype=tf.int64, name='inputs') #len_seq = 40
    y = tf.placeholder(shape=(None,), dtype=tf.float32, name='labels')
    is_training = tf.placeholder_with_default(False, shape = [], name='is_training')
    doc_seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='doc_seq_length')
    sen_seq_length = tf.placeholder(shape=(None,), dtype=tf.int64, name='sen_seq_length')
    X_embed = tf.placeholder(shape=(None, doc_size, sen_size, emb_size), dtype=tf.float32, name='inputs')
    rank_pair = tf.placeholder(tf.float32, shape=(None, None), name = 'rank_pair')

with tf.device('/gpu:1'):

    with tf.name_scope('sen_rnn'):
        X_embed_reshape = tf.reshape(X_embed, [-1, sen_size, emb_size])
        sen_rnn_outputs, sen_rnn_states = bi_rnn(X_embed_reshape, n_hidden = 50, seq_len = sen_seq_length, n_layer = 2, 
                                         is_train = is_training, 
                                         keep_prob = 0.5, scope = 'sen_rnn_block')

    with tf.name_scope('sen_attn'):
        sen_atten_out, sen_atten_w = atten_layer_1(sen_rnn_outputs, atten_size = 50, n_layer = 2, l2reg = 0.001, 
                                                      seq_len = sen_seq_length, use_mask = True, scope = 'sen_attn_block')
        # Dropout
        #sen_atten_out_drop = tf.layers.dropout(sen_atten_out, rate = 1-0.5, training = is_training) # tf.nn.dropout


    #with tf.name_scope('sen_stack'):
        #sen_outs = stack_layer(sen_atten_out, sen_atten_w, sen_rnn_states, X_embed_reshape, scope = 'sen_stack_block')
        #sen_outs = stack_layer(sen_atten_out, sen_rnn_states, scope = 'sen_stack_block')
        #sen_outs_drop = tf.layers.dropout(sen_outs, rate = 1-0.5, training = is_training)

with tf.device('/gpu:2'):
    with tf.name_scope('doc_rnn'):
        doc_inputs = tf.reshape(sen_atten_out, [-1, doc_size, sen_atten_out.shape[1]])
        #doc_inputs = tf.reshape(sen_outs, [-1, doc_size, sen_outs.shape[1]])
        doc_rnn_outputs, doc_rnn_states = bi_rnn(doc_inputs, n_hidden = 50, seq_len = doc_seq_length, n_layer = 2, 
                                         is_train = is_training, 
                                         keep_prob = 0.5, scope = 'doc_rnn_block')

    with tf.name_scope('doc_attn'):
        doc_atten_out, doc_atten_w = atten_layer_1(doc_rnn_outputs, atten_size = 50, n_layer = 2, l2reg = 0.001, 
                                                      seq_len = doc_seq_length, use_mask = True, scope = 'doc_attn_block')
        # Dropout
        #doc_atten_out_drop = tf.layers.dropout(doc_atten_out, rate = 1-0.5, training = is_training) # tf.nn.dropout


    #with tf.name_scope('doc_stack'):
        #doc_outs = stack_layer(doc_atten_out, doc_atten_w, doc_rnn_states, doc_inputs, scope = 'doc_stack_block')
        #doc_outs = stack_layer(doc_atten_out, doc_rnn_states, scope = 'doc_stack_block')
        #doc_outs_drop = tf.layers.dropout(doc_outs, rate = 1-0.5, training = is_training)

    with tf.name_scope('attn_stack'):    
        doc_emb = stack_attn(X_embed_reshape, sen_atten_w, doc_atten_w, scope = None) # [None, doc]


with tf.device('/gpu:3'):
    with tf.name_scope('final_out'):
        #doc_emb_dense = tf.layers.dense(doc_emb, dense_n, activation = None)
        #final_outputs = tf.concat([doc_outs, doc_emb], axis = -1)
        final_outputs = doc_emb
        final_outputs_dense = tf.layers.dense(final_outputs, dense_n, activation = tf.nn.tanh)
        final_outputs_drop = tf.layers.dropout(final_outputs_dense, rate = 1-0.5, training = is_training)

    with tf.name_scope('logits'):
        optimizer, logits, cost = reg_train_op(final_outputs_drop, y, is_train = is_training, 
                                               ac_fn = tf.nn.tanh, lr = 1e-3, l2reg = 0.01, n_output = 1)
        rank_cor = my_corr(rank_pair)
        init, saver = initializer()

    with tf.name_scope('prediction'):
        y_pred = reg_predict_op(logits)        


#g_2 = tf.Graph()    
## prepare embedding
#with tf.device('/cpu:0'):
#with tf.name_scope('embedding'):
#doc_seq_len, sen_seq_len = get_batch_seq(X, doc_len = 15, sen_len = 20)
#X_embed = get_batch_emb(X, doc_len = 15, sen_len = 20)


# In[ ]:





# In[ ]:





# In[ ]:


print(X_embed, '\n', X_embed_reshape)
print('\n\n')
print(sen_atten_out, '\n', sen_atten_w, '\n')
print('\n\n')
print(doc_inputs, '\n')
print(doc_atten_out, '\n', doc_atten_w, '\n')
print('\n\n')
print(doc_emb, '\n', final_outputs, '\n', final_outputs_dense)
print('\n\n')
print(logits,'\n',  cost)


# In[ ]:





# In[ ]:


epochs = 20

loss_record = {}
#acc_record = {}
rank_record = {}
batch_size = 16
#with tf.Session(config=session_conf) as sess:
with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    sess.run(init)

    #saver.restore(sess, './saved_model/best_model')
    min_loss = np.inf

    for epoch in range(epochs + 1):
        loss_train = 0
        loss_test = 0
        #acc_train = 0
        #acc_test = 0
        rank_train = 0
        #rank_test = 0

        #print("epoch: {}\t".format(epoch), end="")

        # training
        n_batch = len(X_train)//batch_size
        for _ in tqdm(range(n_batch), total = n_batch):
            X_batch, y_batch = get_b_t(X_train, y_train, batch_size)
            #input_ids, input_mask, segment_ids = get_batch_inputs(X_batch, di, doc_len = doc_size, sen_len = sen_size)
            #X_batch_emb = get_batch_emb(input_ids, input_mask, segment_ids, layer_n = 4)
            doc_seq_len, sen_seq_len = get_batch_seq(X_batch, doc_len = doc_size, sen_len = sen_size)
            X_batch_emb = get_batch_emb(X_batch, doc_len = doc_size, sen_len = sen_size)
            _, loss, sen_alpha, doc_alpha, y_pred_val = sess.run([optimizer, cost, sen_atten_w, doc_atten_w, y_pred],
                                          feed_dict={X_embed: X_batch_emb, y: y_batch, 
                                                     doc_seq_length: doc_seq_len, 
                                                     sen_seq_length: sen_seq_len, is_training: True})  

            #acc_train += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            rank_val = spearmanr(np.squeeze(y_pred_val), y_batch)[0]
            rank_train += rank_val
            loss_train += loss
        #acc_train /= n_batch
        rank_train /= n_batch
        loss_train /= n_batch

        #if (epoch + 1) % (epochs//10) == 0:
        #if (epoch + 0) % 10 == 0:
            #print('Epoch:', '%d' % (epoch + 0), 'cost =', '{:.6f}'.format(loss))

        # testing
        y_pred_vals = []
        n_batch = len(X_test)//batch_size
        for i in tqdm(range(n_batch), total = n_batch):
            if i == n_batch - 1:
                X_batch = X_test[i*batch_size:]
                y_batch = y_test[i*batch_size:]
            else:
                X_batch = X_test[i*batch_size:(i+1)*batch_size]
                y_batch = y_test[i*batch_size:(i+1)*batch_size]
            #X_batch, y_batch = get_b_t(X_test_new, y_test_new, batch_size)
            doc_seq_len, sen_seq_len = get_batch_seq(X_batch, doc_len = doc_size, sen_len = sen_size)
            X_batch_emb = get_batch_emb(X_batch, doc_len = doc_size, sen_len = sen_size)
            loss, sen_alpha, doc_alpha, y_pred_val = sess.run([cost, sen_atten_w, doc_atten_w, y_pred],
                                          feed_dict={X_embed: X_batch_emb, y: y_batch, 
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
        rank_test = spearmanr(y_pred_vals, y_test)[0]

        loss_record[epoch] = [loss_train, loss_test]
        #acc_record[epoch] = [acc_train, acc_test]
        rank_record[epoch] = [rank_train, rank_test]

        if epoch % 1 == 0:
            print("epoch: {}\t".format(epoch), end="")
            print("loss: {:.5f}, val_loss: {:.5f}, train_rank: {:.5f}, test_rank: {:.5f}".format(
                loss_train, loss_test, rank_train, rank_test))


        if loss_test < min_loss:
            saver.save(sess, '/ids/datasets/chinesenlp/textfast/saved_model_b/best_model')
            min_loss = loss_test

    saver.save(sess,  '/ids/datasets/chinesenlp/textfast/saved_model_b/final_model_b')


# In[ ]:





# In[ ]:





# In[ ]:


idx = 7
s = ''
for ele in X_test_batch[idx]:
    s = s + idx_to_word[ele] + ' '
print(s)   
print(y_test_batch[idx], y_pred[idx])
fig = plt.figure(figsize=(20, 5)) 
ax = fig.add_subplot(1, 1, 1)
cax = ax.matshow(alpha[idx][:, np.newaxis].T, cmap='viridis')
fig.colorbar(cax, orientation = 'horizontal', shrink = 0.2)
ax.set_xticks(np.arange(40))
ax.tick_params(length=0)
ax.set_xticklabels([k for k in s.split()], fontdict={'fontsize': 12}, rotation=90)
ax.set_yticklabels([''])

#ax.set_xticklabels([k for k in tests[0]], fontdict={'fontsize': 14}, rotation=90)
#ax.set_yticklabels([''] + ['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6'], fontdict={'fontsize': 14})
plt.show()


# In[ ]:





# In[ ]:




