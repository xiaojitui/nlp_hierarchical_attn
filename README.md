# nlp_hierarchical_attn

Hierarchical Attention Networks to analyze English and Chinese documents. 
<br><br>
### Basic Algorithm:
The repo provides hierarchical attention networks to analyze documents in English and in Chinese. The basic algorithm is:
- (1) pre-process text data
- (2) get word embeddings (for English) or character embeddings (for Chinese) based on BERT (preferred) or Glove. 
- (3) 1st hierarchy: from word/char embeddings, use Bi-LSTM + Attention layers to get sentence embeddings and word/char attentions (and sentence sentiment scores*)
- (4) 2nd hierarchy: from sentence level, use Bi-LSTM + Attention layers to get document embeddings and sentence attentions

*note: this is optional, the sentence scores can be concatenated to sentence embedding, as the input for the next hierarchy.

The final target of the models can be document classification (e.g. predict if a review is positive/negtive) or regression (e.g. predict sale prices based on news and reports) problems. You can use the sample scripts and modify final layers of the models, or build your own models on the top of the final outputs. 
<br><br>
### Variants of hierarchical attention networks
The repo provides three different networks of using hierarchical structure and embeddings.
- (1) As described in 'Basic Algorithm', one way is to use word/char embeddings to get sentence level, then document level embeddings
- (2) another way is to use the embedding of the first token to represent the whole sentence, the go to document level
- (3) another way is to cut sentences into 'phrases' (instead of words/chars), use the embedding of the first token to represent the phrase, then go to sentence level, then go to document level. 

Check 'chinese_version' and 'english_version' folders for details 
<br><br>
### Embeddings:
- If use BERT to get embeddings, check the repo 'nlp_bert_embedding' for details.
- If use Glove to get embeddings, need to download embedding files 'glove.840B.300d.zip' (or other dimension based on your requirments) 
- If use other pre-trained embeddings, please modify the file paths in the scripts
<br><br>
### Datasets:
English datasets are put in './sample/en' folder; Chinese datasets are put in './sample/ch' folder.
The instruction to download sample datasets is provided in the 'notes.txt' file in the './sample/en (or ch)' folder. 

There are two types of raw data: doc_level and sen_level.
- (1) doc_level: each single data is one document, which contains multiple sentences. 
- (2) sen_level: each single data is just one sentence (or say, there is only one sentence in each document.)
