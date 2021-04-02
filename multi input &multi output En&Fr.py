#!/usr/bin/env python
# coding: utf-8

# # TASK #1: IMPORT LIBRARIES AND DATASETS

# In[1]:


from collections import Counter
import operator
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, TimeDistributed, RepeatVector, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[2]:


get_ipython().system('pip install --upgrade tensorflow-gpu==2.0')


# In[3]:


# load the data
df_english = pd.read_csv('small_vocab_en.csv', sep = '/t', names = ['english'])
df_french = pd.read_csv('small_vocab_fr.csv', sep = '/t', names = ['french'])


# In[4]:


df_english.info()


# In[5]:


df_french.info()


# 
# - Concatenate both dataframes and indicate how many records are present
# - Print out the following: "Total English Records = xx, Total French Records = xx"

# In[6]:


df=pd.concat([df_english,df_french],axis=1)


# In[7]:


df


# In[8]:


print('Total English Records = {}'.format(len(df['english'])))
print('Total French Records = {}'.format(len(df['french'])))


# # TASK #2: PERFORM DATA CLEANING

# In[9]:


# download nltk packages
nltk.download('punkt')

# download stopwords
nltk.download("stopwords")


# In[10]:


# function to remove punctuations
def remove_punc(x):
    return re.sub('[!#?,.:";]', '', x)


# In[11]:


df['french'] = df['french'].apply(remove_punc)
df['english'] = df['english'].apply(remove_punc)


# In[12]:


english_words = []
french_words  = []


# 
# - How many unique words are available in the english and french dictionairies?
# 
# 
# 
# 

# In[13]:


def get_unique_words(x,word_list):
    for word in x.split():
        if word not in word_list:
            word_list.append(word)
            
df['english'].apply(lambda x: get_unique_words(x,english_words))
        
df['french'].apply(lambda x: get_unique_words(x,french_words))


# In[14]:


total_english_words=len(english_words)
total_english_words


# In[15]:


# number of unique words in french
total_french_words=len(french_words)
total_french_words


# # TASK #3: VISUALIZE CLEANED UP DATASET

# In[16]:


# Obtain list of all words in the dataset
words = []
for i in df['english']:
    for word in i.split():
        words.append(word)
    
words


# In[17]:


# Obtain the total count of words
english_words_counts = Counter(words)
english_words_counts


# In[18]:


# sort the dictionary by values
english_words_counts = sorted(english_words_counts.items(), key = operator.itemgetter(1), reverse = True)


# In[19]:


english_words_counts


# In[20]:


# append the values to a list for visualization purposes
english_words = []
english_counts = []
for i in range(len(english_words_counts)):
    english_words.append(english_words_counts[i][0])
    english_counts.append(english_words_counts[i][1])


# In[21]:


english_words


# In[22]:


english_counts


# In[23]:


# Plot barplot using plotly 
fig = px.bar(x = english_words, y = english_counts)
fig.show()


# In[24]:


# plot the word cloud for text that is Real
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000, width = 1600, height = 800 ).generate(" ".join(df.english))
plt.imshow(wc, interpolation = 'bilinear')


# In[25]:


df.english[0]
nltk.word_tokenize(df.english[0])


# In[26]:


# Maximum length (number of words) per document. We will need it later for embeddings
maxlen_english = -1
for doc in df.english:
    tokens = nltk.word_tokenize(doc)
    if(maxlen_english < len(tokens)):
        maxlen_english = len(tokens)
print("The maximum number of words in any document = ", maxlen_english)


# 
# - Perform similar data visualizations but for the french language instead
# - What are the top 3 common french words?!
# - What is the maximum number of words in any french document?

# In[27]:


words = []
for i in df['french']:
    for word in i.split():
        words.append(word)
words


# In[28]:


french_words_counts = Counter(words)
french_words_counts


# sort the dictionary by values
french_words_counts = sorted(french_words_counts.items(), key = operator.itemgetter(1), reverse = True)

french_words_counts


# In[29]:


# append the values to a list for visuaization purpose
french_words = []
french_counts = []
for i in range(len(french_words_counts)):
    french_words.append(french_words_counts[i][0])
    french_counts.append(french_words_counts[i][1])


# In[ ]:





# In[30]:


fig = px.bar(x = french_words, y = french_counts)
fig.show()



# In[31]:


# plot the word cloud for French
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df.french))
plt.imshow(wc, interpolation = 'bilinear')



# In[32]:


# Maximum length (number of words) per document. We will need it later for embeddings
maxlen_french = -1
for doc in df.french:
    tokens = nltk.word_tokenize(doc)
    if(maxlen_french < len(tokens)):
        maxlen_french = len(tokens)
print("The maximum number of words in any document = ", maxlen_french)


# # TASK #4: PREPARE THE DATA BY PERFORMING TOKENIZATION AND PADDING

# In[33]:


def tokenize_and_pad(x, maxlen):
    
  #  a tokenier to tokenize the words and create sequences of tokenized words
    tokenizer = Tokenizer(char_level = False)
    tokenizer.fit_on_texts(x)
    sequences = tokenizer.texts_to_sequences(x)
    padded = pad_sequences(sequences, maxlen = maxlen, padding = 'post')
    return tokenizer, sequences, padded


# In[34]:


# tokenize and padding to the data 
x_tokenizer, x_sequences, x_padded = tokenize_and_pad(df.english, maxlen_english)
y_tokenizer, y_sequences, y_padded = tokenize_and_pad(df.french,  maxlen_french)


# In[35]:


# Total vocab size, since we added padding we add 1 to the total word count
english_vocab_size = total_english_words + 1
print("Complete English Vocab Size:", english_vocab_size)


# In[36]:


# Total vocab size, since we added padding we add 1 to the total word count
french_vocab_size = total_french_words + 1
print("Complete French Vocab Size:", french_vocab_size)


# In[37]:


print("The tokenized version for document\n", df.english[-1:].item(),"\n is : ", x_padded[-1:])


# In[38]:


print("The tokenized version for document\n", df.french[-1:].item(),"\n is : ", y_padded[-1:])


# In[39]:


# function to obtain the text from padded variables
def pad_to_text(padded, tokenizer):

    id_to_word = {id: word for word, id in tokenizer.word_index.items()}
    id_to_word[0] = ''

    return ' '.join([id_to_word[j] for j in padded])


# In[40]:


pad_to_text(y_padded[0], y_tokenizer)


# In[41]:


# Train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_padded, y_padded, test_size = 0.1)


# 
# - Change the padding length so that both english and french have the same length

# In[42]:


# tokenize and padding to the data 
x_tokenizer, x_sequences, x_padded = tokenize_and_pad(df.english, maxlen_french)
y_tokenizer, y_sequences, y_padded = tokenize_and_pad(df.french,  maxlen_french)


# # TASK #5: UNDERSTAND THE THEORY AND INTUITION BEHIND RECURRENT NEURAL NETWORKS AND LSTM
# 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #6: UNDERSTAND THE INTUITION BEHIND LONG SHORT TERM MEMORY (LSTM) NETWORKS

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK #7: BUILD AND TRAIN THE MODEL 

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[47]:


# Sequential Model
model = Sequential()
# embedding layer
model.add(Embedding(english_vocab_size, 256, input_length = maxlen_english, mask_zero = True))
# encoder
model.add(LSTM(256))
# decoder
# repeatvector repeats the input for the desired number of times to change
# 2D-array to 3D array. For example: (1,256) to (1,23,256)
model.add(RepeatVector(maxlen_french))
model.add(LSTM(256, return_sequences= True ))
model.add(TimeDistributed(Dense(french_vocab_size, activation ='softmax')))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[48]:


# change the shape of target from 2D to 3D
y_train = np.expand_dims(y_train, axis = 2)
y_train.shape


# In[49]:


# train the model
model.fit(x_train, y_train, batch_size=512, validation_split= 0.25, epochs=10)


# In[50]:


# save the model
model.save("weights.h5")


# 
# - Train the model with different embedding output dimension and comment on model performance during training
# 

# In[ ]:





# # TASK #8: ASSESS TRAINED MODEL PERFORMANCE
# 

# In[ ]:


# function to make prediction
def prediction(x, x_tokenizer = x_tokenizer, y_tokenizer = y_tokenizer):
    predictions = model.predict(x)[0]
    id_to_word = {id: word for word, id in y_tokenizer.word_index.items()}
    id_to_word[0] = ''
    return ' '.join([id_to_word[j] for j in np.argmax(predictions,1)])


# In[ ]:


for i in range(5):
    print('Original English word - {}\n'.format(pad_to_text(x_test[i], x_tokenizer)))
    print('Original French word - {}\n'.format(pad_to_text(y_test[i], y_tokenizer)))
    print('Predicted French word - {}\n\n\n\n'.format(prediction(x_test[i:i+1])))


# # CONGRATULATIONS!
