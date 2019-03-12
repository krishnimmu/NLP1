##importing the [packages]()
import re

# importing pandas module
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

%matplotlib inline

#read the data from the csv file
df = pd.read_csv("C:\\Users\\chandra\\Downloads\\NLP_Task\\nlpdata.txt",sep=",,,",header=None ,names=['question','type'])

#shows the top rows in the file
df.head()

#shows the number of rows and columns
df.shape

#string specifying the set of characters to be removed
df['type']=df['type'].str.strip()

#List unique values in the df['type'] column
df['type'].unique()

#number of independent values that can vary in an analysis without breaking any constraints
df['question'].values

#converting a string to lowercase
df['question'] = df['question'].apply(lambda x: x.lower())
#applying string operations to pandas dataframe
df['question'] = df['question'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

#give me all the input data I will take care of splitting between test and validation
VALIDATION_SPLIT=0.20

###Naive Bayes with tfidf vectorizer
##multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
from sklearn.naive_bayes import MultinomialNB

# organize imports
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
class StemTokenizer(object):
    def __init__(self):
        self.ignore_set = {'footnote', 'nietzsche', 'plato', 'mr.'}

    def __call__(self, doc):
        words = []
        for word in word_tokenize(doc):
            word = word.lower()
            w = wn.morphy(word)
            if w and len(w) > 1 and w not in self.ignore_set:
                words.append(w)
        return words
    	stemmer = SnowballStemmer('english').stem
def stem_tokenize(text):
    return [stemmer(i) for i in word_tokenize(text)]
	
####Using Count Vectorizer

vectorizer = CountVectorizer(analyzer='word',lowercase=True,tokenizer=stem_tokenize)
X_train = vectorizer.fit_transform(df.question.values)
with open('vectorizer.pk', 'wb') as fin:
    pkl.dump(vectorizer, fin)
labels = data['type']

# split the data into a training set and a validation set
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * X_train.shape[0])

x_train = X_train[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = X_train[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
clf = MultinomialNB()
clf.fit(x_train,y_train)

# evaluate the model of test data
preds = clf.predict(x_val)
print(classification_report(preds,y_val))
print("Accuracy :",clf.score(x_val,y_val))
example=vectorizer.transform(["What time does the train leave"])
clf.predict(example)


###Using TF-IDF (though bad choice for short sequences or corpus)

tf_vectorizer = TfidfVectorizer(analyzer='word',lowercase=True,tokenizer=stem_tokenize)
X_train = tf_vectorizer.fit_transform(df.question.values)
with open('tf_vectorizer.pk', 'wb') as fin:
    pkl.dump(tf_vectorizer, fin)

labels = data['type']

# split the data into a training set and a validation set
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * X_train.shape[0])

x_train = X_train[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = X_train[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

clf = MultinomialNB()
clf.fit(x_train,y_train)

# evaluate the model of test data
preds = clf.predict(x_val)
print(classification_report(preds,y_val))
print("Accuracy :",clf.score(x_val,y_val))

example=tf_vectorizer.transform(["What time does the train leave"])
clf.predict(example)


####LSTM

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH=30

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re

data=df.copy()

print(data['type'].value_counts())

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, split=' ')
tokenizer.fit_on_texts(data['question'].values)
X = tokenizer.texts_to_sequences(data['question'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Y = data['type']
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Y)
Y=le.transform(Y) 
labels = to_categorical(np.asarray(Y))
print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', labels.shape)


# split the data into a training set and a validation set
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = X[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open('E:/Projects/Word2Vec/glove.42B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM=300

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
       
	   # words not found in embedding index will be all-zeros.
	    embedding_matrix[i] = embedding_vector

from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

mbed_dim = 300
lstm_out = 196

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(lstm_out, dropout_U=0.25, dropout_W=0.25))
model.add(Dense(5,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_val, y_val))

example = tokenizer.texts_to_sequences(["What time does the train leave"])
example = pad_sequences(example, maxlen=MAX_SEQUENCE_LENGTH)

le.inverse_transform(np.argmax(model.predict(example)))
		  

