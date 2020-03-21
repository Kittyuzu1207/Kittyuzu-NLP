# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 00:56:32 2019

@author: Kitty
"""
#nltk prediction
import nltk
import jieba
import codecs
f=codecs.open('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/2/news.txt','r','utf8')
news=[eval(i) for i in f.readlines()]
f.close()

#model1:Naive Bayes  看词与类的共现关系
f=open('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/2/train.txt','r')
train=[i.split() for i in f.readlines()]
f.close()

f=open('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/2/test.txt','r')
test=[i.split() for i in f.readlines()]
f.close()

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]  
    return stopwords 
stoplist=stopwordslist('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/2/stopword.txt')

def word_depart(data):  #分词 data是str的list
    material=[]
    for i in range(len(data)):
        if type(data[i])!=float:  #排除是nan
            tmp=data[i].replace("\n", "").strip()
            tmp=jieba.cut(tmp,cut_all=False)
            out=[]
            for word in tmp:
                if word not in stoplist and word != '\t'and word!=' ':  #去停用词
                    out.append(word)  
            material.append(out)
    return material


def preprocessing(train,news):
    raw=[]
    label=[]
    news_dict={}
    for i in range(len(news)):
        news_dict[news[i]['id']]=news[i]['content']
    for i in range(len(train)):
        outstr=' '
        if train[i][0]=='+1':
            tmp=train[i][1].split(',')
            label.append('+1')
            for j in range(len(tmp)):
                outstr+=news_dict[int(tmp[j])]
            raw.append(' '+outstr)
        else:
            tmp=train[i][1].split(',')
            label.append('-1')
            for j in range(len(tmp)):
                outstr+=news_dict[int(tmp[j])]
            raw.append(' '+outstr)
    return raw,label


x_train,y_train=preprocessing(train,news)  
x_test,y_test=preprocessing(test,news)

def word_depart2(data):  #分词 data是str的list
    out=[]
    for i in range(len(data)):
        if type(data[i])!=float:  #排除是nan
            tmp=data[i].replace("\n", "").strip()
            tmp=jieba.cut(tmp,cut_all=False)
            for word in tmp:
                if word not in stoplist and word != '\t'and word!=' ':  #去停用词
                    out.append(word)  
    return out

out=[]
for i in range(len(x_train)):
    if type(x_train[i])!=float:  #排除是nan
        tmp=x_train[i].replace("\n", "").strip()
        tmp=jieba.cut(tmp,cut_all=False)
        for word in tmp:
            if word not in stoplist and word != '\t'and word!=' ':  #去停用词
                out.append(word)  
                
all_words = nltk.FreqDist(w for w in out)
most_common_word = all_words.most_common(1000)
most_common=[]
for i in range(len(most_common_word)):
    most_common.append(most_common_word[i][0])



def doc_feature(doc):
    doc_words = set(doc)
    feature = {}
    for word in most_common:
        feature[word] = (word in doc_words)
    return feature
    
docs_train=[(word_depart2([x_train[i]]),y_train[i])
           for i in range(len(x_train))]

docs_test=[(word_depart2([x_test[i]]),y_test[i])
           for i in range(len(x_test))]


train_set = nltk.apply_features(doc_feature, docs_train)
test_set = nltk.apply_features(doc_feature, docs_test)



#classifier1 = nltk.NaiveBayesClassifier.train(train_set)
#result6=[]
#for i in range(len(test_set)):
#    result6.append(classifier1.classify(test_set[i][0]))
#
#f=open('result6.txt','w')
#for i in range(len(result6)):
#    #f.write(y_predict[i]+'\t'+test[i][1]+'\n')
#    f.write(result6[i]+'\n')
#f.close()




'''Desicion Tree'''
classifier2 = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier2, test_set))
result2=[]
for i in range(len(test_set)):
    result2.append(classifier2.classify(test_set[i][0]))

f=open('result35.txt','w')
for i in range(len(result2)):
    #f.write(y_predict[i]+'\t'+test[i][1]+'\n')
    f.write(result2[i]+'\n')
f.close()

'''SVM'''
from sklearn.svm import LinearSVC
classifier4 = nltk.classify.SklearnClassifier(LinearSVC())
classifier4.train(train_set)

result4=[]
for i in range(len(test_set)):
    result4.append(classifier4.classify(test_set[i][0]))

f=open('result45.txt','w')
for i in range(len(result4)):
    #f.write(y_predict[i]+'\t'+test[i][1]+'\n')
    f.write(result4[i]+'\n')
f.close()

'''Logistic'''

from sklearn.linear_model import LogisticRegression
classifier6=nltk.classify.SklearnClassifier(LogisticRegression())
classifier6.train(train_set)
result5=[]
for i in range(len(test_set)):
    result5.append(classifier6.classify(test_set[i][0]))

f=open('result55.txt','w')
for i in range(len(result5)):
    #f.write(y_predict[i]+'\t'+test[i][1]+'\n')
    f.write(result5[i]+'\n')
f.close()
