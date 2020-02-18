# Chinese Version
The scripts here provide three hierarchical attention networks to analyze Chinese document.

(1) 'train_cn_doc_bt.py'
- each dataset is one document with multiple sentences.
- use BERT to get character embeddings.
- hierarchical structure is from characters to sentences to document


(2) 'train_cn_doc_bt_allsen.py'
- each dataset is one document with multiple sentences.
- use BERT to get the embedding of the first token, to represent the whole sentence
- hierarchical structure is from sentences to document


(3) 'train_cn_doc_bt_phrases.py'
- each dataset is one document with multiple sentences.
- cut sentences into 'phrases' (instead of characters)
- use BERT to get the embedding of the first token, to represent the phrase
- hierarchical structure is from phrases to sentences to document


In the 'character_to_words_first' folder, the scripts provide another way to process Chinese text. The algorithm is to group Chinese characters into words first, then use pre-trained Chinese word embeddings as input of the model. 
