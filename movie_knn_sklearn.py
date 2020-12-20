# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
import random
from collections import Counter

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
 
# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    doc=doc.replace('<br />',' ') 
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    for word in y:
        if len(word)<=2:
            del y[y.index(word)]
    return y


path = "movie.txt"

train_clean_sentences = []
y_train=[]
test_clean_sentences = []
y_test=[]
fp = open(path,'r')

ds=[]
for row in fp:
    ds.append([row[:-2],int(row[-2])])

for i in range(len(ds)):
    #ds[i][0]=' '.join(clean(ds[i][0]))
    cleaned= clean(ds[i][0])
    cleaned = ' '.join(cleaned)
    ds[i][0]=cleaned

random.shuffle(ds)
split=int(len(ds)*0.8)
for i in range(split):
    train_clean_sentences.append(ds[i][0])
    y_train.append(ds[i][1])

for i in range(split,len(ds)):
    test_clean_sentences.append(ds[i][0])
    y_test.append(ds[i][1])

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)

# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=17)
modelknn.fit(X,y_train)



Test = vectorizer.transform(test_clean_sentences)

#true_test_labels = ['Cricket','AI','Chemistry']
predicted_labels_knn = modelknn.predict(Test)
 
print ("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")

ap_pp=0
ap_pn=0
an_pp=0
an_pn=0
acc=0

for i in range(len(y_test)):
               if predicted_labels_knn[i]==y_test[i]:
                   acc+=1
                   if(y_test[i]==1):
                       ap_pp+=1
                   else:
                        an_pn+=1
               else:
                    if(y_test[i]==1):
                       ap_pn+=1
                    else:
                        an_pp+=1
pct= 100/len(y_test)
accuracy= acc* pct
print('   Pos     Neg')
print('  ',ap_pp*pct,'   ',an_pp*pct)
print('  ',ap_pn*pct,'   ',an_pn*pct)
print('Accuracy: ', accuracy,'%')
