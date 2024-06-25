# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import plotly.graph_objs as go 
import os
import re
import nltk
import plotly.offline as py 
py.init_notebook_mode(connected=True)
import plotly.tools as tls  
from collections import Counter                        
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from nltk.corpus import stopwords


# %%
df = pd.read_csv("amazon_alexa.tsv",sep='\t')
df.head()

# %%
len(df)

# %%
# HOW MANY ALEXA PRODUCTS ARE THERE?
df['variation'].nunique()

# %%
# HOW MANY COMMENTS HAVE BEEN MADE ABOUT WHICH PRODUCT? 
df['variation'].value_counts()

# %%
df.describe()

# %%
df.groupby('rating').describe()

# %%
df['length'] = df['verified_reviews'].apply(len)
df.head()

# %%
# WHICH ALEXA PRODUCT GETS THE HIGHEST VOTES? 
df.groupby('variation').agg({"rating":"mean"}).sort_values('rating', ascending = False).head()

# %%
# HOW MANY POSITIVE FEEDBACKS ARE THERE? 
# 1 = Positive , 0 = negative
df["feedback"].value_counts().head()

# %%
#HOW MANY POSITIVE AND NEGATIVE FEEDBACKS ARE THERE ?
trace0 = go.Bar(
            x = df[df["feedback"]== 1]["feedback"].value_counts().index.values,
            y = df[df["feedback"]== 1]["feedback"].value_counts().values,
            name='Positive Feedback')

trace1 = go.Bar(
            x = df[df["feedback"]== 0]["feedback"].value_counts().index.values,
            y = df[df["feedback"]== 0]["feedback"].value_counts().values,
            name='Negative Feedback')


data = [trace0, trace1]
layout = go.Layout(yaxis=dict(title='Count'),
                   xaxis=dict(title='Feedback'),title='Feedback Distribution')

fig = go.Figure(data=data, layout=layout)
fig.data[0].marker.line.width = 4
fig.data[0].marker.line.color = "black"
fig.data[1].marker.line.width = 4
fig.data[1].marker.line.color = "black"
py.iplot(fig)

# %%
# AVERAGE VOTES OF THE PRODUCTS 
rating = df.groupby('variation').agg({"rating":"mean"})
rating['variation'] = rating.index
rating.reset_index(drop=True)

trace = go.Bar(x=rating['variation'], y=rating['rating'])

layout = go.Layout(yaxis=dict(title='Average Rating'),
                   xaxis=dict(title='Alexa Product'),title='Product - Avarage Rating Distribution')

fig = go.Figure(data=trace, layout=layout)
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"
py.iplot(fig)

# %%
# HOW MUCH WAS USED FROM WHICH VOTES?
trace = go.Bar(
            x = df["rating"].value_counts().index.values,
            y = df["rating"].value_counts().values,
            name='Quantity')

layout = go.Layout(yaxis=dict(title='Quantity'),
                   xaxis=dict(title='Ratings'),title='# of Votes Quantity')

fig = go.Figure(data=trace, layout=layout)
fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"
py.iplot(fig)

# %%
# importing the dataset
dataset = pd.read_csv('amazon_alexa.tsv', delimiter = '\t', quoting = 3) 

# %%
nltk.download('stopwords')

# %%
###CLEANING THE TEXT

# %%
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# %%
# It is a process of normalization
text2 = "Kiss kissed kisses know knowing last lasting"
stemmer = PorterStemmer()
Norm_Word= stemmer.stem(text2)
Tokens = text2.split()
" ".join(stemmer.stem(token) for token in Tokens)

# %%
STOPWORDS = set(stopwords.words('english'))
corpus=[]
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]', ' ', Data['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    stemmer = PorterStemmer()
    review = [stemmer.stem(token) for token in review if not token in STOPWORDS]
    #contain all words that are not in stopwords dictionary
    review=' '.join(review)
    corpus.append(review)
corpus

# %%
words = []
for i in range(0,len(corpus)):
    words = words + (re.findall(r'\w+', corpus[i]))# words cantain all the words in the dataset
words

# %%
len(words)

# %%
from collections import Counter
words_counts = Counter(words)
print(words_counts)

# %%
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)
most_common_words

# %%
most_commmom_wordList = []
most_commmom_CountList = []
for x, y in most_common_words:
    most_commmom_wordList.append(x)
    most_commmom_CountList.append(y)

# %%
import seaborn as sns
plt.figure(figsize=(20,18))
plot = sns.barplot(x=most_commmom_wordList[0:20], y=most_commmom_CountList[0:20])
plt.ylabel('Word Count', fontsize=20)
plt.xticks(rotation=40, fontsize=20)
plt.title('Most Frequently Used Words', fontsize=20)
plt.show()


# %%
# creating the Bag of words Model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,4].values

# %%
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# %%
import xgboost as xgb

# %%
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
       
    


# %%
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# %%
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# %%
cm

# %%
from sklearn.metrics import f1_score

# Calculate the F-score
f_score = f1_score(y_test, y_pred)

# Show the score
print("F-score: %", f_score*100)


# %%



