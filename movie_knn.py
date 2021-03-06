#Importing libraries
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
import random
from collections import Counter
from math import log


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


#this method returns important words from a sentence as list
def getwords(sentence):
    w = sentence.split(" ")
    #add bigrams
    w= w + [w[i]+' '+w[i+1] for i in range(len(w)-1)]
    w= list(set(w))
    return w


path = "movie.txt"

train_clean_sentences = []
y_train=np.array([])
test_clean_sentences = []
y_test=np.array([])
fp = open(path,'r')

ds=[]
for row in fp:
    ds.append([row[:-2],int(row[-2])])

for i in range(len(ds)):
    cleaned= clean(ds[i][0])
    cleaned = ' '.join(cleaned)
    ds[i][0]=cleaned

random.shuffle(ds)

poslines=[]
neglines=[]
for i in ds:
    if i[1]==1:
        poslines.append(i[0])  
    else:
        neglines.append(i[0])

possplit=int(len(poslines)*0.8)
negsplit=int(len(neglines)*0.8)

train_clean_sentences= [(x,1) for x in poslines[:possplit]] + [(x,0) for x in neglines[:negsplit]]
#y_train= [1]*possplit + [0]*negsplit
y_train=np.append(y_train,[[1]*possplit + [0]*negsplit])

test_clean_sentences= [(x,1) for x in poslines[possplit:]] + [(x,0) for x in neglines[negsplit:]]
#y_test= [1]*(len(poslines)-possplit) + [0]*(len(neglines)-negsplit)
y_test=np.append(y_test,[[1]*(len(poslines)-possplit) + [0]*(len(neglines)-negsplit)])


train_len= len(train_clean_sentences)
freq={}
trainfeatures= []
for line,label in train_clean_sentences:
    words= getwords(line)
    for word in words: 
        freq[word]= freq.get(word, 0) + 1
    trainfeatures.append((words,label))

predicted_labels_knn=np.array([])

for testline,testlabel in test_clean_sentences:
    testwords= getwords(testline)
    results=[]
    for trainwords, trainlabel in trainfeatures:
        #find all words in common between these two sentences
        commonwords= [x for x in trainwords if x in testwords]
        score= 0.0		
        for word in commonwords:
            score += log(train_len/freq[word])
        results.append((score, trainlabel))

    results.sort(reverse=True)
    toplab= [x[1] for x in results[:5]]
    numones= toplab.count(1)
    numnegones= toplab.count(0)
	
    if numnegones> numones:
        predicted_labels_knn=np.append(predicted_labels_knn,0)
    else:
        predicted_labels_knn=np.append(predicted_labels_knn,1)
 
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
