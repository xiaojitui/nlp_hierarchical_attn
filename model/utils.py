#!/usr/bin/env python
# coding: utf-8

# In[23]:


#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from tqdm import tqdm


# In[ ]:





# ## read config

# In[1]:


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


# In[ ]:





# ## prepare dataset

# In[4]:


def get_train_test(X, y, seed = None, train_ratio = 0.9):
    
    idx = np.arange(len(X))
    
    if seed is not None:
        np.random.seed(seed)
        
    np.random.shuffle(idx) 
    
    idx_1 = idx[:int(train_ratio*len(X))] 
    idx_2 = idx[int(train_ratio*len(X)):]
    
    X_train = np.array([X[k] for k in idx_1])
    y_train = np.array([y[k] for k in idx_1])
    X_test = np.array([X[k] for k in idx_2])
    y_test = np.array([y[k] for k in idx_2])
    
    return X_train, y_train, X_test, y_test


# In[5]:


def get_batch_data(X, y, batch_size, seed = None):
    
    if seed is not None:
        np.random.seed(seed)
    
    idx = np.random.choice(len(X), batch_size, replace=False)
    y = np.array(y)
    X_batch = X[idx]
    y_batch = y[idx]
    
    return X_batch, y_batch


# In[3]:


def get_batch_test(X, y, iter_n, batch_size):    
    n_batch = len(X)//batch_size
    if iter_n == n_batch - 1:
        X_batch = X[iter_n*batch_size:]
        y_batch = y[iter_n*batch_size:]
    else:
        X_batch = X[iter_n*batch_size:(iter_n+1)*batch_size]
        y_batch = y[iter_n*batch_size:(iter_n+1)*batch_size]
    return X_batch, y_batch


# In[ ]:





# In[ ]:





# ## text statistics 

# In[16]:


# get word counts of each sentence
# sentences = [sen1, sen2, sen3,....]
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


# In[17]:


# get sentence counts of each document
# get word counts of each sentence
# docs = [ [sen1, sen2, sen3,....],  [sen1, sen2, sen3,....],  [sen1, sen2, sen3,....]...]
# sentences = [sen1, sen2, sen3,....]

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


# In[ ]:





# In[ ]:





# ## prepare sen, words

# In[19]:


def split_sentence(txt):
    sents = re.split(r'\n|\s|;|；|。|，|\.|,|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|、|╱|∥', txt)
    sents = [c for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
    sents = list(filter(None, sents))
    return sents


# In[20]:


def split_sentence_1(txt):
    sents = re.split(r'\n|。|\.|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|╱|∥|<sssss>', txt)
    sents = [c.strip() for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
    sents = list(filter(None, sents))
    return sents


# In[21]:


def replace_contr(sents, contractions):
    sent_clean = []
    for sent in sents:
        for word in sent.split():
            if word.lower() in contractions:
                sent = sent.replace(word, contractions[word.lower()])
                
        if len(sent.split()) >1:
            sent_clean.append(sent)
    return np.array(sent_clean)


# In[8]:


def clean_sentence(sents):
    sents_clean = []
    for sent in sents:
        sent = sent.strip("'").strip('"').strip()
        sent = sent.split()
        sent_clean = ''
        i = 0
        while i < len(sent)-1:
            if "'" in sent[i+1]:
                ele = sent[i] + sent[i+1]
                i += 2
            else:
                ele = sent[i]
                i += 1
            sent_clean = sent_clean + ele + ' '
        if i == len(sent) - 1:
            sent_clean = sent_clean + sent[i]
        
        sent_clean = sent_clean.replace('-lrb- ', '(').replace(' -rrb-', ')')
        sent_clean = sent_clean.replace("`` ", "'").replace(" ''", "'")
        sents_clean.append(sent_clean)
    return sents_clean


# In[ ]:





# In[22]:


## combine all
## workflow: split, clean, replace contractions

def prepare_sen(docs, contractions):
    
    X = []
    for text in docs:
        sents = split_sentence_1(text) ##################### use: _1 or not
        sents_clean = replace_contr(clean_sentence(sents), contractions)
        X.append(sents_clean)
        
    return X


# In[5]:


def prepare_words(docs, contractions):
    
    X = []
    for text in docs:
        
        sents = replace_contr(clean_sentence(text), contractions)
        split_sents = []
        for sent in sents:
            words = split_sentence(sent) ##################### use: _1 or not
            split_sents.append(words)
        X.append(split_sents)
        
    return X


# In[ ]:





# In[ ]:





# ## voc, token, emb

# In[9]:


def load_embedding(path):

    words = []
    idx = 0
    word_to_idx = {}
    vectors = []
    with open(path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word_to_idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    emb_dim = len(vectors[0])
    # add OOV and PADDING
    words.extend(['UNK', 'PAD'])
    word_to_idx[words[-2]] = len(word_to_idx)
    word_to_idx[words[-1]] = len(word_to_idx)
    vectors = np.concatenate([vectors, np.zeros((2, emb_dim))]).astype(np.float32)
    return word_to_idx, words, vectors


# In[24]:


# build vocabulary
def build_voc(sentences):
    words = ' '.join(sentences).split()
    words = [k.lower() for k in words]
    voc = list(set(words))
    
    return voc


# In[25]:


def token(voc):
    word_to_idx = {w:i for i, w in enumerate(voc)}
    word_to_idx['UNK'] = len(voc)
    word_to_idx['PAD'] = len(voc) + 1
    idx_to_word = {i:w for w, i in word_to_idx.items()}
    
    return word_to_idx, idx_to_word


# In[30]:


# token single sentence
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


# In[31]:


# token a batch of sentences == token a doc
def token_sens(sentences, sentence_size, word_to_idx):
    sen_idx_bath = []
    for sen in sentences:
        sen_idx = token_sen(sen, sentence_size, word_to_idx)
        sen_idx_bath.append(sen_idx)
        
    return np.array(sen_idx_bath)


# In[32]:


# token a batch of docs
def token_docs(docs, doc_size, sen_size, word_to_idx):
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


# In[ ]:





# ## seq

# In[ ]:





# In[ ]:





# In[35]:


# X_bacth = [sent1, sent2, sent3,....]

def get_sen_batch_seq_1(X_batch, pad_token = 400001):
    
    batch_seq_len = []
    for x in X_batch:
        if pad_token in x:
            cur_seq = list(x).index(pad_token)
        else:
            cur_seq = len(x)
        batch_seq_len.append(cur_seq)
    
    batch_seq_len = np.array(batch_seq_len)
    return batch_seq_len


# In[36]:


def get_doc_batch_seq_1(X_batch, pad_token = 400001):
    
    doc_seq_len = []
    sen_seq_len = []
    for doc in X_batch:
        if pad_token in doc.T[0]:
            cur_doc_seq = list(doc.T[0]).index(pad_token)
        else:
            cur_doc_seq = len(doc)
        doc_seq_len.append(cur_doc_seq)
        
        for sen in doc:
            if pad_token in sen:
                cur_sen_seq = list(sen).index(pad_token)
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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## backup

# In[ ]:


# do 
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




