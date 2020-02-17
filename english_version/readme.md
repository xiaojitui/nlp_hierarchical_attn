The scripts here provide mutiple hierarchical attention networks to analyze English document


(1) 'train_en_doc.py'
each dataset is one document with multiple sentences.
use Glove to get word embeddings.
hierarchical structure is from words to sentences to document

(2) 'train_en_doc_bt.py'
each dataset is one document with multiple sentences.
use BERT to get word embeddings.
hierarchical structure is from words to sentences to document

(3) 'train_en_doc_bt_allsen.py'
each dataset is one document with multiple sentences.
use BERT to get the embedding of the first token, to represent the whole sentence
hierarchical structure is from sentences to document

(4) 'train_en_doc_bt_phrases.py'
each dataset is one document with multiple sentences.
cut sentences into 'phrases' (instead of words)
use BERT to get the embedding of the first token, to represent the phrase
hierarchical structure is from phrases to sentences to document





(5) 'train_en_sen.py'
each dataset is one sentence (i.e. only one sentence in each document)
use Glove to get word embeddings.
hierarchical structure is from words to sentence

(6) 'train_en_sen_bt.py'
each dataset is one sentence (i.e. only one sentence in each document)
use BERT to get word embeddings.
hierarchical structure is from words to sentence

(7) 'train_en_sen_bt_phrases.py'
each dataset is one sentence (i.e. only one sentence in each document)
cut sentences into 'phrases' (instead of words)
use BERT to get the embedding of the first token, to represent the phrase
hierarchical structure is from phrases to sentence

