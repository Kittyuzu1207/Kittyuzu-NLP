# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:32:40 2019

@author: Administrator
"""
import math
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



class bayes_model():
    def __init__(self):
        pass
    def word_depart(self,data):  #data是x_train的语料list
        material=[]
        for i in range(len(data)):
            if type(data[i])!=float:  #排除是nan
                tmp=data[i].replace("\n", "").strip()
                tmp=jieba.cut(tmp,cut_all=False)
                out=[]
                for word in tmp:
                    if word not in stoplist and word != '\t'and word!=' ':  
                        out.append(word)  
                material.append(out)
        return material
    def prob_c(self,x_train,y_train):
        pos=0
        neg=0
        pos_raw=[]
        neg_raw=[]
        for i in range(len(y_train)):
            if y_train[i]=='+1':
                pos+=1
                pos_raw.append(x_train[i])
            else:
                neg+=1
                neg_raw.append(x_train[i])
        pos_prob=pos/(pos+neg)
        neg_prob=neg/(pos+neg)
        self.pos_prob=(pos_prob+0.1)/(1+0.2)  #对类别smoothing
        self.neg_prob=(neg_prob+0.1)/(1+0.2)
        self.pos_raw=pos_raw
        self.neg_raw=neg_raw
    def word_class_dict(self,x_train,y_train):
        self.prob_c(x_train,y_train)
        words=[]
        temp=self.word_depart(self.pos_raw)
        for i in range(len(temp)):
            words+=temp[i]
        temp=self.word_depart(self.neg_raw)
        for i in range(len(temp)):
            words+=temp[i] 
       
        words=list(set(words))  #训练语料的总词典
        self.words=words
        vocab_len=len(words)
        word_dict1={}
        word_dict2={}
        all_count1=0
        all_count2=0
        for w in words:
            count1=0
            count2=0
            for i in range(len(self.pos_raw)):
                if w in self.pos_raw[i]:
                    count1+=1
                    all_count1+=1  #分母求和归一化项一
            for j in range(len(self.neg_raw)):
                if w in self.neg_raw[j]:
                    count2+=1
                    all_count2+=1
            word_dict1[w]=count1   #正向  
            word_dict2[w]=count2    #负向
        for w in word_dict1:
            word_dict1[w]=(word_dict1[w]+1)/(all_count1+vocab_len)  #Laplace smoothing
        for w in word_dict2:
            word_dict2[w]=(word_dict2[w]+1)/(all_count2+vocab_len)
        self.vocab_len=vocab_len
        self.all_count1=all_count1
        self.all_count2=all_count2
        self.word_dict1=word_dict1
        self.word_dict2=word_dict2        
        
    def train(self,x_train,y_train):
        self.word_class_dict(x_train,y_train)
        
    def predict_senten(self,sentence):   #判断sentence是哪一个class的
        sen=self.word_depart([sentence])[0]
        prob1=self.pos_prob  
        prob2=self.neg_prob
        prob1=math.log(prob1)       #防止概率相乘太小下溢
        prob2=math.log(prob2)
        for w in sen:
            if w in self.word_dict1:
                prob1+=math.log(self.word_dict1[w])

            if w in self.word_dict2:
                prob2+=math.log(self.word_dict2[w])


        if prob1>prob2:  #可用带参数的max函数实现argmax
            return '+1'
        if prob1<prob2:
            return '-1'
        
    def predict(self,x_test):
        label=[]
        for sen in x_test:
            label.append(self.predict_senten(sen))
        return label
    
    

    
clf=bayes_model()
x_test,y_test=preprocessing(test,news)
clf.train(x_train,y_train)


y_predict=clf.predict(x_test)

f=open('E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/2/result24.txt','w')
for i in range(len(test)):
    #f.write(y_predict[i]+'\t'+test[i][1]+'\n')
    f.write(y_predict[i]+'\n')
f.close()


#count=0
#for i in range(200):
#    if y_predict[i]==y_test[i]:
#        count+=1
#    print(y_predict[i])
#print(count)