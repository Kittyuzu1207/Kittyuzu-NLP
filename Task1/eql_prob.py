# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 00:07:55 2019

@author: Kitty
"""

'''法3：equally'''
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import reuters
freq=nltk.FreqDist([w.lower() for w in reuters.words()])  #利用nltk的reuters语料库 频数c(w)
freq2=nltk.FreqDist(nltk.bigrams([w.lower() for w in reuters.words()]))  
import time
import string
import re
import readfile
import nonword

def read_file(path):
    return readfile.read_file(path)

def readvoc(path):
    return readfile.readvoc(path)

path='E:\\!!!!!!!!!!study\\学习\\大四上\\NLP\\NLP intro\\homework\\1\\spelling-correction\\Assignment1\\'
testdata=read_file(path+'testdata.txt')
testdata.columns=(['id','error num','senten']) 
vocab=readvoc(path+'vocab.txt')
path2='E:\\!!!!!!!!!!study\\学习\\大四上\\NLP\\NLP intro\\homework\\1\\'
transition=read_file(path2+'transition.txt')
transition.columns=['char','count']
transition['x'],transition['w']=transition['char'].str.split('|').str
del transition['char']
count=0
transition['count']=transition['count'].astype('int')
p_edit1=readfile.prob_edit1()

def generate_candidates(data):  #data是很多行句子组成的数据  testdata['senten']
    can_list=[]
    for i in range(len(data)):
        words=data[i].split()
        can=[]
        for j in range(len(words)):
            can.append(candidates(words[j]))
        can_list.append(can)
    return can_list
def edit1(word):
    return nonword.edit1(word)    
def candidates(word):
    return set(list(known(edit1(word)))+list(known([word])))  #要把这个词本身加进去
def known(words):
    return set(w for w in words if w in vocab)
def prob_w(w1,w2): 
   V=len(freq2)
   p=(freq2[(w1.lower(),w2.lower())]+1)/(freq[w1.lower()]+V)
   return p
def prob_trans(w,x,alpha):
    if w==x:
        return alpha
    else:
        return (1-alpha)*p_edit1  #出错的概率乘上编辑距离为1的概率
    
def prob(w1,w2,w3,x,alpha):
    return prob_w(w1,w2)*prob_w(w2,w3)*prob_trans(w2,x,alpha)
    

def prob_last(w1,w2,x,alpha):
    return prob_w(w1,w2)*prob_trans(w2,x,alpha)

def prob_first(w2,w3,x,alpha):
    return prob_w(w2,w3)*prob_trans(w2,x,alpha)

def correct(word,w1,w3,alpha):
    #要考虑到是否是第一个或者最后一个词
    if(len(candidates(word))>0):
        return max(candidates(word),key=lambda item:prob(w1,item,w3,word,alpha))
    else:
        return word
def correct_last(word,w1,alpha):#有上文没有下文 /最后一个词
    if(len(candidates(word))>0):
        return max(candidates(word),key=lambda item:prob_last(w1,item,word,alpha))
    else:
        return word
def correct_first(word,w3,alpha):#有下文没上文 /第一个词
    if(len(candidates(word))>0):
        return max(candidates(word),key=lambda item:prob_first(item,w3,word,alpha))
    else:
        return word
def main():
    start=time.clock()
    result_list=[]
    for i in range(1000):
        can=[]
        words=nltk.word_tokenize(testdata['senten'][i])
        for j in range(len(words)):
            p=re.findall(r"[,.?!']",words[j])
            n=re.findall(r"[0-9]+",words[j])
            if(len(p)==0 and len(n)==0):
                if j==0:  #开头
                    can.append(correct_first(words[j],words[j+1],0.95))
                elif j==len(words)-1:   #结尾
                    can.append(correct_last(words[j],words[j-1],0.95))
                else:  #中间
                    can.append(correct(words[j],words[j-1],words[j+1],0.95))   
            else:
                can.append(words[j])
        result_list.append(can)
        print("sen "+str(i)+'have been processed')
    print(time.clock()-start)
    result=[]
for i in range(len(result_list)):
    words=result_list[i]
    sen=result_list[i][0]
    for j in range(1,len(words)):
        if words[j] in ',.?!' or len(re.findall(r"[a-zA-Z]*'[a-zA-Z]*",words[j]))>0:
            sen=sen+words[j]
        else:
            sen=sen+' '+words[j]
    result.append(sen)
    f = open("E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/result_equal.txt",'w')
    for i in range(len(result)):
        f.write(str(i+1)+'\t'+result[i]+'\n') 
    f.close() 
    anspath='E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/ans.txt'
    resultpath='E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/result_equal.txt'
    ansfile=open(anspath,'r')
    resultfile=open(resultpath,'r')
    count=0
    for i in range(1000):
        ansline=ansfile.readline().split('\t')[1]
        ansset=set(nltk.word_tokenize(ansline))
        resultline=resultfile.readline().split('\t')[1]
        resultset=set(nltk.word_tokenize(resultline))
        if ansset==resultset:
            count+=1
    print(count)
    print("Accuracy is : %.2f%%" % (count*1.00/10))