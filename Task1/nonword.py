# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:07:26 2019

@author: Kitty
"""
#纠正non-word错误
#error detection:遍历当前句子，没有在词典里出现的为error
#编写函数计算某一个词和词典中某个词的编辑距离,为error词计算编辑距离为1的词语列表
#基于channel model,选择概率最大的w作为最终的拼写建议
#通过对语料库计数、平滑等处理建立language model，可得到P(w)
#基于大量的错误,计算转移矩阵和转移概率p(x|w)
import pandas as pd
import string
import re
import time
import nltk
import readfile
from nltk.corpus import brown  #reuters
freq=nltk.FreqDist([w.lower() for w in brown.words()])  #利用nltk的reuters语料库 频数c(w)
freq2=nltk.FreqDist(nltk.bigrams([w.lower() for w in brown.words()]))  #计算bigram 频数c(w1,w2)
#from nltk.corpus import reuters
#freq=nltk.FreqDist([w.lower() for w in reuters.words()])  #利用nltk的reuters语料库 频数c(w)
#freq2=nltk.FreqDist(nltk.bigrams([w.lower() for w in reuters.words()]))  
testdata,ans,vocab=readfile.load_non()

def nonword_detection(sen):  #检测一个句子中的non-word，返回词语本身，索引和是否有's
    words=sen.split()
    err=[]
    flag=[] #记录错误单词的索引和是否为所有格
    for i in range(len(words)):
        word=words[i].strip(string.punctuation)
        if word not in vocab:
            if (len(word.split("'"))<2 and word+'.' not in vocab) or (len(word.split("'"))>1 and word.split("'")[0] not in vocab and word.split("'")[0][:-1] not in vocab) and word!='':
            #发现连写形式如Japan's don't不在词典里，需要进行修正  
            #isn't don't doesn't didn't wouldn't can't 
                err.append(word.split("'")[0])
                if(len(word.split("'"))>1):
                    flag.append([i,1])
                else:
                    flag.append([i,0])
    return err,flag,words


#def edit_distance(str1,str2):  #计算两个单词的编辑距离
#    if len(str1) == 0:
#        return len(str2)
#    elif len(str2) == 0:
#        return len(str1)
#    elif str1 == str2:
#        return 0
#    if str1[len(str1)-1] == str2[len(str2)-1]:
#        d = 0
#    else:
#        d = 1  
#    if len(str1)>1 and len(str2)>1:
#        if(str1[-1]==str2[-2] and str1[-2]==str2[-1] and str1[-1]!=str1[-2]):
#            dd=1
#        else:
#            dd=100
#    else:
#        dd=100 #如果不是相邻两个交换就可以的，使dd大，排除这种操作
#    return min(edit_distance(str1, str2[:-1]) + 1,
#               edit_distance(str1[:-1], str2) + 1,
#               edit_distance(str1[:-1], str2[:-1]) + d,edit_distance(str1[:-2],str2[:-2])+dd)


#def candidate_list(word,dic): #返回词典中距离当前word编辑距离最近（为1）的候选词列表
#    candidate=[]
#    for w in dic:     #计算量好大，这样真的没做错么。。。真的跑不动
#        if(edit_distance(word,w)==1 or edit_distance(word,w)==2):  #仅保留编辑距离为1和2的  
#            candidate.append(w)
#    return candidate

def edit1(word):
    #与'word'的编辑距离为1的全部结果
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])     for i in range(len(word) + 1)]
    deletes    = [L + R[1:]                for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:]  for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]            for L, R in splits for c in letters]
    inserts    = [L + c + R                for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)    

def edit2(word):
    return (e2 for e1 in edit1(word) for e2 in edit1(e1))

def known(words):
    return set(w for w in words if w in vocab)

def candidates(word):
    return (known([word]) or known(edit1(word)) or [word])
     #return (known([word]) or known(edit1(word)) or known(edit2(word)) or [word])


#通过计数和add-one平滑，建立language model
def prob_w(w1,w2):
   #V 是bi-gram 的type数
   #p=(count(w1,w2)+1)/(count(w1)+V)  
   V=len(freq2)
   p=(freq2[(w1.lower(),w2.lower())]+1)/(freq[w1.lower()]+V)
   return p


def prob_trans(w,x,alpha):
    #要识别w到x是哪一种edit
    #有错的概率*这种错占这个字母所有错误的比例
    return (1-alpha)*word_trans(x,w)
    

#识别w到x是什么edit w是正确单词 x是实际单词
def Trans(str2,str1):
    return readfile.Trans(str2,str1)

def word_trans(w,x):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(w[:i], w[i:])     for i in range(len(w) + 1)]
    deletes    = [L + R[1:]                for L, R in splits if R]
    if x in deletes:
        for j in range(len(x)):
           if(w[j]!=x[j]):
               if j==0:
                   return Trans(w[j:j+2],x[j])   #开头
               else:
                   return Trans(w[j-1:j+1],x[j-1])  #中间
        return Trans(w[-2:],x[len(x)-1])          #结尾
    inserts    = [L + c + R                for L, R in splits for c in letters]
    if x in inserts:
        for j in range(len(w)):
            if(w[j]!=x[j]):
                if j==0:
                    return Trans(w[j],x[j:j+2])  #开头
                else:
                    return Trans(w[j-1],x[j-1:j+1])   #中间
        return Trans(w[len(w)-1],x[-2:])     #结尾
    transposes = [L + R[1] + R[0] + R[2:]  for L, R in splits if len(R) > 1]
    if x in transposes:
        for j in range(len(x)-1):
            if(x[j]==w[j+1] and x[j+1]==w[j]):
                return Trans(w[j:j+2],x[j:j+2])
    replaces   = [L + c + R[1:]            for L, R in splits for c in letters]
    if x in replaces:
        for j in range(len(x)):
            if(x[j]!=w[j]):
                return Trans(w[j],x[j])
    return 0 

    

def prob(w1,w2,w3,x,alpha):
    return prob_w(w1,w2)*prob_w(w2,w3)*prob_trans(w2,x,alpha)
    

def prob_last(w1,w2,x,alpha):
    return prob_w(w1,w2)*prob_trans(w2,x,alpha)

def prob_first(w2,w3,x,alpha):
    return prob_w(w2,w3)*prob_trans(w2,x,alpha)

#误差模型，计算条件概率/转移概率 p(x|w)

#找候选词中p(x|w)p(w)最大的作为建议词 在Python中，带key的max()函数即可实现argmax的功能
def correct(word,w1,w3,alpha):
    #要考虑到是否是第一个或者最后一个词
    return max(candidates(word),key=lambda item:prob(w1,item,w3,word,alpha))
def correct_last(word,w1,alpha):#有上文没有下文 /最后一个词
    return max(candidates(word),key=lambda item:prob_last(w1,item,word,alpha))
def correct_first(word,w3,alpha):#有下文没上文 /第一个词
    return max(candidates(word),key=lambda item:prob_first(item,w3,word,alpha))


def main():
    start = time.clock()
    errlist=[]
    wordlist=[]
    flaglist=[]
    for i in range(len(testdata)):
        err,flag,word=nonword_detection(testdata.iloc[i]['senten'])
        errlist.append(err)
        wordlist.append(word)
        flaglist.append(flag)
        
    corrlist=[]
    for i in range(len(errlist)):
        corr=[]
        if len(errlist[i])>0:  #存在non-word error
            for j in range(len(errlist[i])):
                if flaglist[i][j][0]!=0 and flaglist[i][j][0]!=len(wordlist[i])-1:
                    corr.append(correct(errlist[i][j],wordlist[i][flaglist[i][j][0]-1],wordlist[i][flaglist[i][j][0]+1],0.9))
                elif flaglist[i][j][0]!=0:
                    corr.append(correct_last(errlist[i][j],wordlist[i][flaglist[i][j][0]-1],0.9))
                else:
                    corr.append(correct_first(errlist[i][j],wordlist[i][flaglist[i][j][0]+1],0.9))
        corrlist.append(corr)
    print(time.clock()-start)   
    #把修正的词复原到原来的句子里
    result=pd.DataFrame(testdata['senten'])
    for i in range(len(result)):
        for j in range(len(errlist[i])):
            #flaglist[i][j][0]是位置，flaglist[i][j][1]是标记是否为所有格,好像不用索引和标记所有格，直接replace
            result.iloc[i]['senten']=result.iloc[i]['senten'].replace(errlist[i][j],corrlist[i][j])
    #保存结果到文档
    f = open("result6.txt",'w')
    for i in range(len(result)):
        f.write(str(i+1)+'\t'+result.iloc[i]['senten']) 
    f.close()   
    dt = time.clock() - start
    print(dt)

