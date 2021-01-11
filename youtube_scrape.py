# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:09:41 2020

@author: Sumit Dembla
"""

from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd

import nltk
from nltk import FreqDist

nltk.download('stopwords')

import numpy as np
import re
import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline




'''
YouTubeTranscriptApi is a python API which allows you to get the
transcripts/subtitles for a given YouTube video
writen by jdepoix
https://github.com/jdepoix/youtube-transcript-api.git
'''

with open('VideoID.txt') as f:
    ID_list = f.readlines()


ls = []
for ID in ID_list:
    try: 
        caption = YouTubeTranscriptApi.get_transcript(video_id=ID, languages=['en'])
        caption1 = [i['text'] for i in caption]
        caption_new = ''.join(caption1)
        # file_name = ID.replace('-', "")
        # file_name = file_name.strip('\n')
        ls.append(caption_new)
        
        
        # file_name = file_name.strip('\n') + '.txt'
        # f = open(file_name, "w")
        # f.write(caption_new)
        # f.close()
    except:
        ls.append('NULL')
        # file_name = 'Null_' + ID.replace('-', "")
        # file_name = file_name.strip('\n') + '.txt'
        # f = open(file_name, "w")
        # f.close()

df = pd.DataFrame(ls)
df.to_csv('subtitles.csv')

import pickle
with open('list_1.txt', 'wb') as fp:
    pickle.dump(ls, fp)

with open ('list_1.txt', 'rb') as fp:
    list_1 = pickle.load(fp)


df = pd.read_excel('SASGF_YouTube_with_Metadata.xlsx')
df['transcript'] = list_1

df.head()

# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()


freq_words(df['transcript'])

df['transcript'] = df['transcript'].str.replace("[^a-zA-Z#]", " ")

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new

# remove short words (length < 3)
df['transcript'] = df['transcript'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# remove stopwords from the text
trans = [remove_stopwords(r.split()) for r in df['transcript']]

# make entire text lowercase
trans = [r.lower() for r in trans]

freq_words(trans, 35)

nlp = spacy.load('en_core_web_sm')


#nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output
   

    
tokenized_trans = pd.Series(trans).apply(lambda x: x.split())
print(tokenized_trans[1])    

trans_2 = lemmatization(tokenized_trans)
print(trans_2[1]) # print lemmatized review    

trans_3 = []
for i in range(len(trans_2)):
    trans_3.append(' '.join(trans_2[i]))

df['transcript'] = trans_3

freq_words(df['transcript'], 35)

dictionary = corpora.Dictionary(trans_2)

doc_term_matrix = [dictionary.doc2bow(rev) for rev in trans_2]

# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                chunksize=1000, passes=50)


lda_model.print_topics()

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis



all_words = ' '.join([text for text in df['transcript']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()








