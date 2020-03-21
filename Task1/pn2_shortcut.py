# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 08:29:00 2019

@author: Kitty
"""
'''法2 PN's shortcut'''
import pandas as pd
import string
import re
import nltk
from nltk.corpus import brown
freq=nltk.FreqDist([w.lower() for w in brown.words()])#利用nltk的布朗语料库来计算p(w)
freq2=nltk.FreqDist(nltk.bigrams([w.lower() for w in brown.words()]))
#from nltk.corpus import reuters
#freq=nltk.FreqDist([w.lower() for w in reuters.words()])  #利用nltk的reuters语料库 频数c(w)
#freq2=nltk.FreqDist(nltk.bigrams([w.lower() for w in reuters.words()]))  
import time
import string
import re
import readfile
testdata,ans,vocab=readfile.load_non()

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
#    return (known([word]) or known(edit1(word)) or known(edit2(word)) or [word])

def prob_w(w1,w2):  
   V=len(freq2)
   p=(freq2[(w1.lower(),w2.lower())]+1)/(freq[w1.lower()]+V)
   return p

def prob(w1,w2,w3,x,alpha):
    return prob_w(w1,w2)*prob_w(w2,w3)
    

def prob_last(w1,w2,x,alpha):
    return prob_w(w1,w2)

def prob_first(w2,w3,x,alpha):
    return prob_w(w2,w3)
def correct(word,w1,w3,alpha):
    #要考虑到是否是第一个或者最后一个词
    return max(candidates(word),key=lambda item:prob(w1,item,w3,word,alpha))
def correct_last(word,w1,alpha):#有上文没有下文 /最后一个词
    return max(candidates(word),key=lambda item:prob_last(w1,item,word,alpha))
def correct_first(word,w3,alpha):#有下文没上文 /第一个词
    return max(candidates(word),key=lambda item:prob_first(item,w3,word,alpha))

def main():
    start=time.clock()
    #直接对已经correct 过non-word的语料进行处理
    testdata=readfile.read_file('E:\\!!!!!!!!!!study\\学习\\大四上\\NLP\\NLP intro\\homework\\1\\spelling-correction\\Assignment1\\'+'result6.txt')
    testdata.columns=(['id','senten'])
    result_list=[]
    for i in range(len(testdata)):
        can=[]
        words=nltk.word_tokenize(testdata['senten'][i])
        for j in range(len(words)):
            p=re.findall(r"[,.?!']",words[j])
            if(len(p)==0):
                if j==0:  #开头
                    can.append(correct_first(words[j],words[j+1],0.9))
                elif j==len(words)-1:   #结尾
                    can.append(correct_last(words[j],words[j-1],0.9))
                else:  #中间
                    can.append(correct(words[j],words[j-1],words[j+1],0.9))   
            else:
                can.append(words[j])
        result_list.append(can)
        print("sen "+str(i)+'have been processed')
    print(time.clock()-start)
    #把句子复原
    result=[]
    for i in range(len(testdata)):
        words=result_list[i]
        sen=result_list[i][0]
        for j in range(1,len(words)):
            if words[j] in ',.?!':
                sen=sen+words[j]
            else:
                sen=sen+' '+words[j]
        result.append(sen)
    f = open("E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/result_pn2.txt",'w')
    for i in range(len(result)):
        f.write(str(i+1)+'\t'+result[i]+'\n') 
    f.close() 
    #结果
    anspath='E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/ans.txt'
    resultpath='E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/result_pn2.txt'
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
    
if __name__=='__main__':
    main()
