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





# In[ ]:





# ## load data

# In[2]:


path = './samples/en/sen_level'
os.listdir(path)


# In[3]:


train_data = pd.read_csv(os.path.join(path, 'train_binary_sent.csv'))
dev_data = pd.read_csv(os.path.join(path, 'dev_binary_sent.csv'))
test_data = pd.read_csv(os.path.join(path, 'test_binary_sent.csv'))

train_data.shape, test_data.shape, dev_data.shape


# In[4]:


train_data.head()


# In[5]:


train_data.label.value_counts()


# In[6]:


for i in range(10):
    print(i, '\t', train_data.iloc[i, 1])


# In[ ]:





# In[ ]:





# ## preprocess

# In[7]:


with open('./model/contractions.pkl', 'rb') as f:
    contractions = pickle.load(f)


# In[8]:


X_train_raw = train_data['sentence'].values
y_train = train_data['label'].values
X_test_raw = test_data['sentence'].values
y_test = test_data['label'].values

X_train_raw.shape, y_train.shape, X_test_raw.shape, y_test.shape


# In[ ]:





# In[9]:


X_train = []

for x in tqdm(X_train_raw, total = len(X_train_raw)):
    x = utils.clean_sentence([x])
    x = utils.replace_contr(x, contractions)
    #X_train.append(list(x)) # usually this is x[0], here, use list(x), to make X_train a 2D list
    x = utils.split_document(x[0])
    x, _ = utils.refine_document(x) 
    X_train.append(x[0])  


X_test = []

for x in tqdm(X_test_raw, total = len(X_test_raw)):
    x = utils.clean_sentence([x])
    x = utils.replace_contr(x, contractions)
    #X_test.append(list(x)) # usually this is x[0], here, use list(x), to make X_train a 2D list
    x = utils.split_document(x[0])
    x, _ = utils.refine_document(x) 
    X_test.append(x[0])
    
len(X_train), len(y_train), len(X_test), len(y_test)


# In[ ]:





# In[10]:


X_train[:5], X_train_raw[:5]


# In[11]:


X_test[:5], X_test_raw[:5]


# In[ ]:





# ## study sample statistics

# In[12]:


idx = np.random.choice(6920)
X_train[idx], y_train[idx], X_train_raw[idx]


# In[13]:


idx = np.random.choice(1821)
X_test[idx], y_test[idx], X_test_raw[idx]


# In[14]:


# X_train is 1D list = [sent1, sen2, sen3.....]
doc_cut_val, sen_cut_val, doc_len, sen_len = utils.get_doc_stat(X_train + X_test, doc_cut_thres = 0.90, sen_cut_thres = 0.90)
# usually just use X_train + X_test
doc_cut_val, sen_cut_val


# In[15]:


doc_cut_val, sen_cut_val, doc_len, sen_len = utils.get_doc_stat(X_train + X_test, doc_cut_thres = 0.95, sen_cut_thres = 0.95)
# usually just use X_train + X_test
doc_cut_val, sen_cut_val


# In[16]:


# choose doc_len = 4, sen_len = 20


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## prepare voc, token, emb

# In[17]:


bert_path = './bert/checkpoint/en'
bert_config_file = './bert/checkpoint/en/bert_config.json'
bert_vocab_file = './bert/checkpoint/en/vocab.txt'
init_checkpoint = './bert/checkpoint/en/bert_model.ckpt'
graph_file = './bert/tmp_en/graph' # where to put temporary graph
select_layers = [-4, -3, -2, -1] 
#select_layers = [0] 
doc_len = 4
sen_len = 20

batch_size = 32


# In[ ]:





# In[18]:


tokenizer, estimator = embedding_bert.prepare_bert(bert_vocab_file, bert_config_file, init_checkpoint, sen_len, 
                                                   select_layers, batch_size, graph_file)


# In[19]:


### prepare_bert(bert_vocab_file, bert_config_file, init_checkpoint, sen_len, select_layers,  batch_size, graph_file)


# In[20]:


### test


# In[19]:


X_train[:3]


# In[20]:


a = embedding_bert.get_batch_emb(X_train[:3], doc_len, sen_len, tokenizer, estimator)
b, c = embedding_bert.get_batch_seq(X_train[:3], doc_len, sen_len, tokenizer, tol = 2)


# In[ ]:





# In[21]:


a.shape, b.shape, c.shape


# In[22]:


b


# In[23]:


c


# In[24]:


batch_id = 0
a[batch_id][0].shape


# In[25]:


xx = a[:, :, 0, :]
xx.shape


# In[26]:


#### test end


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Train section

# In[27]:


#vocab_size = emb.shape[0]
#embedding_size = emb.shape[1]
#pretrained_emb = True
#finetune_emb = False
doc_len = 4
sen_len = 20
keep_prob = 0.5

epochs = 10 #20
batch_size = 64 #64

n_hidden = 50
n_layer = 2

atten_size = 50
l2reg = 0.0 #1e-5
use_mask = True
sen_CLS = False

n_class = 2
lr = 1e-4

savepath = './saved_weights/best_model.ckpt'
finalpath = './saved_weights/final_model.ckpt'


# In[28]:


# generate doc emb
def get_doc_emb(x_emb, doc_attn_w, scope = None):
    
    with tf.variable_scope(scope or 'stack_attn') as scope:
        # X_emb = [None, doc, embedding]
        # doc_attn_w = [None, doc]
        doc_attn_w = tf.expand_dims(doc_attn_w, -1) #[None, doc, 1]
        doc_emb = tf.reduce_sum(doc_attn_w * X_emb, axis = 1) # [None, embedding]
        
    return doc_emb


# In[40]:


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
    


# In[41]:


3072/np.sqrt(3072/2)


# In[42]:


print(X_emb)
print('\n\n')
print(atten_out, '\n', soft_atten_weights)
print('\n\n')
print(doc_emb, '\n', doc_out, '\n', doc_out_1)
print('\n\n')
print(logits,'\n',  cost)


# In[ ]:





# In[43]:


loss_record = {}
acc_record = {}
#with tf.Session(config=session_conf) as sess:
with tf.Session() as sess:
    sess.run(init)
    
    #saver.restore(sess, savepath)
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
            idx = np.random.choice(len(X_train), batch_size, replace = False)
            X_batch = [X_train[k] for k in idx]
            y_batch = [y_train[k] for k in idx]
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
            X_batch, y_batch = utils.get_batch_test(X_test, y_test, i, batch_size)
            X_batch_emb = embedding_bert.get_batch_emb(X_batch, doc_len, sen_len, tokenizer, estimator)
            X_batch_emb = X_batch_emb[:, :, 0, :]
            X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
            loss, acc, atten_w, logits_val = sess.run([cost, accuracy, soft_atten_weights, logits],
                                          feed_dict={X_emb: X_batch_emb, y: y_batch, seq_length: X_batch_seq, is_training:False}) #seq_length: batch_seq_len
            acc_test += acc
            # loss_train = loss_tr * DELTA + loss_train * (1 - DELTA), delta = 0.5??
            loss_test += loss
        acc_test /= n_batch

        loss_record[epoch] = [loss_train, loss_test]
        acc_record[epoch] = [acc_train, acc_test]

        #if epoch ==0 or (epoch + 1) % 5 == 0:
        if epoch >= 0:
            print("epoch: {}\t".format(epoch + 1), end="")
            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
                loss_train, loss_test, acc_train, acc_test), end = "\t")
            print(round(time.time() - t0, 3))


        if loss_test < min_loss:
            saver.save(sess, savepath)
            min_loss = loss_test

    saver.save(sess,  finalpath)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Predict section

# In[49]:


with tf.Session() as sess:
    saver.restore(sess, savepath)
    
    #X_batch, y_batch = utils.get_batch_data(X_test, y_test, 5)
    idx = np.random.choice(len(X_test), 10, replace = False)
    X_batch = [X_test[k] for k in idx]
    y_batch = [y_test[k] for k in idx]
            
    X_batch_emb = embedding_bert.get_batch_emb(X_batch, doc_len, sen_len, tokenizer, estimator)
    X_batch_emb = X_batch_emb[:, :, 0, :]
    X_batch_seq, _ = embedding_bert.get_batch_seq(X_batch, doc_len, sen_len, tokenizer, tol = 2)
    atten_w, y_proba = sess.run([soft_atten_weights, Y_proba], feed_dict={X_emb: X_batch_emb, 
                                                                          seq_length: X_batch_seq, is_training:False})
    
    y_pred = np.argmax(y_proba, axis = 1)


# In[ ]:





# In[50]:


X_batch, y_batch, y_pred


# In[52]:


atten_w[6]


# In[ ]:





# In[ ]:





# In[60]:


X_batch[idx]


# In[56]:


idx = 6
#s = tokenizer.tokenize(X_batch[idx][0])
#s = [' '] + s
s = X_batch[idx]
for i in range(len(s)):
    print(s[i], round(atten_w[idx][i], 3), end = ';')
fig = plt.figure(figsize=(20, 5)) 
ax = fig.add_subplot(1, 1, 1)
cax = ax.matshow(atten_w[idx][:, np.newaxis].T, cmap='viridis')
fig.colorbar(cax, orientation = 'horizontal', shrink = 0.2)
ax.set_xticks(np.arange(doc_len))
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




