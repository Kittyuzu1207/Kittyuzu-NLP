# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:44:41 2019

@author: Kitty
"""

'''法4：迭代概率'''
import pandas as pd
import numpy as np
import nonword
import readfile
import re
import nltk
from nltk.corpus import brown
freq=nltk.FreqDist([w.lower() for w in brown.words()])#利用nltk的布朗语料库来计算p(w)
#from nltk.corpus import reuters
#freq=nltk.FreqDist([w.lower() for w in reuters.words()])  #利用nltk的reuters语料库 频数c(w)
#freq2=nltk.FreqDist(nltk.bigrams([w.lower() for w in reuters.words()]))  
import time
import string
#建立一个字典来记录和更新频率
prob_dict={}   #key是 [edit类型]|
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
    return nonword.edit1(word)

def candidates(word):
    return set(list(known(edit1(word)))+[word])  #要把这个词本身加进去

def known(words):
    return nonword.known(words)

def prob_w(w1,w2):
   return nonword.prob_w(w1,w2)

#不要用单词的频率统计，用edit的，因为单词出现的很少
def update_del(w,x):
    flag=0
    for j in range(len(x)):
           if(w[j]!=x[j]):
               if j==0:
                  prob_dict[w[j:j+2]+'|'+x[j]]+=1   #开头
                  flag=1
               else:
                  prob_dict[w[j-1:j+1]+'|'+x[j-1]]+=1  #中间
                  flag=1
    if(flag==0):
        prob_dict[w[-2:]+'|'+x[len(x)-1]]+=1

def update_ins(w,x):
    flag=0
    for j in range(len(w)):
            if(w[j]!=x[j]):
                if j==0:
                    prob_dict[w[j]+'|'+x[j:j+2]]+=1  #开头 
                    flag=1
                else:
                    prob_dict[w[j-1]+'|'+x[j-1:j+1]]+=1   #中间
                    flag=1
    if(flag==0):
        prob_dict[w[len(w)-1]+'|'+x[-2:]]=1     #结尾

def update_trans(w,x):
    for j in range(len(x)-1):
            if(x[j]==w[j+1] and x[j+1]==w[j]):
                prob_dict[w[j:j+2]+'|'+x[j:j+2]]+=1

def update_rep(w,x):
    for j in range(len(x)):
            if(x[j]!=w[j]):
                prob_dict[w[j]+'|'+x[j]]=1

def update(w,x):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(w[:i], w[i:])     for i in range(len(w) + 1)]
    deletes    = [L + R[1:]                for L, R in splits if R]
    inserts    = [L + c + R                for L, R in splits for c in letters]
    transposes = [L + R[1] + R[0] + R[2:]  for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]            for L, R in splits for c in letters]
    if x in deletes:
        update_del(w,x)
    elif x in inserts:
        update_ins(w,x)
    elif x in transposes:
        update_trans(w,x)
    elif x in replaces:
        update_rep(w,x)
        
    
def edit_trans(w,x):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(w[:i], w[i:])     for i in range(len(w) + 1)]
    deletes    = [L + R[1:]                for L, R in splits if R]
    if x in deletes:
        for j in range(len(x)):
           if(w[j]!=x[j]):
               if j==0:
                   if w[j:j+2]+'|'+x[j] not in prob_dict.keys():
                       prob_dict[w[j:j+2]+'|'+x[j]]=1
                   return float(prob_dict[w[j:j+2]+'|'+x[j]]/sum(prob_dict.values()))
               else:
                   if w[j-1:j+1]+'|'+x[j-1] not in prob_dict.keys():
                       prob_dict[w[j-1:j+1]+'|'+x[j-1]]=1  #中间
                   return float(prob_dict[w[j-1:j+1]+'|'+x[j-1]]/sum(prob_dict.values()))
        if w[-2:]+'|'+x[len(x)-1] not in prob_dict.keys():
            prob_dict[w[-2:]+'|'+x[len(x)-1]]=1          #结尾
        return float(prob_dict[w[-2:]+'|'+x[len(x)-1]]/sum(prob_dict.values()))
       
    inserts    = [L + c + R                for L, R in splits for c in letters]
    if x in inserts:
        for j in range(len(w)):
            if(w[j]!=x[j]):
                if j==0:
                    if w[j]+'|'+x[j:j+2] not in prob_dict.keys():
                        prob_dict[w[j]+'|'+x[j:j+2]]=1  #开头 
                    return float(prob_dict[w[j]+'|'+x[j:j+2]]/sum(prob_dict.values()))
                else:
                    if w[j-1]+'|'+x[j-1:j+1] not in prob_dict.keys():
                        prob_dict[w[j-1]+'|'+x[j-1:j+1]]=1   #中间
                    return float(prob_dict[w[j-1]+'|'+x[j-1:j+1]]/sum(prob_dict.values()))
        if w[len(w)-1]+'|'+x[-2:] not in prob_dict.keys():
            prob_dict[w[len(w)-1]+'|'+x[-2:]]=1     #结尾
        return float(prob_dict[w[len(w)-1]+'|'+x[-2:]]/sum(prob_dict.values()))
    
    transposes = [L + R[1] + R[0] + R[2:]  for L, R in splits if len(R) > 1]
    if x in transposes:
        for j in range(len(x)-1):
            if(x[j]==w[j+1] and x[j+1]==w[j]):
                if w[j:j+2]+'|'+x[j:j+2] not in prob_dict.keys():
                    prob_dict[w[j:j+2]+'|'+x[j:j+2]]=1
                return float(prob_dict[w[j:j+2]+'|'+x[j:j+2]]/sum(prob_dict.values()))
    replaces   = [L + c + R[1:]            for L, R in splits for c in letters]
    if x in replaces:
        for j in range(len(x)):
            if(x[j]!=w[j]):
                if w[j]+'|'+x[j] not in prob_dict.keys():
                    prob_dict[w[j]+'|'+x[j]]=1
                return float(prob_dict[w[j]+'|'+x[j]]/sum(prob_dict.values()))
    return 0 
      
 
def prob_trans(w,x,alpha):
    if w==x:
        return alpha
    else:
        return (1-alpha)*edit_trans(w,x)

def prob(w1,w2,w3,x,alpha):
    return prob_w(w1,w2)*prob_w(w2,w3)*prob_trans(w2,x,alpha)    

def prob_last(w1,w2,x,alpha):
    return prob_w(w1,w2)*prob_trans(w2,x,alpha)

def prob_first(w2,w3,x,alpha):
    return prob_w(w2,w3)*prob_trans(w2,x,alpha)

def correct(word,w1,w3,alpha):
    #要考虑到是否是第一个或者最后一个词
    goal=max(candidates(word),key=lambda item:prob(w1,item,w3,word,alpha))
    if goal!=word:
        update(goal,word)#update编辑频率
    return goal
def correct_last(word,w1,alpha):#有上文没有下文 /最后一个词
    goal=max(candidates(word),key=lambda item:prob_last(w1,item,word,alpha))
    if goal!=word:
        update(goal,word)   #update编辑频率
    return goal
def correct_first(word,w3,alpha):#有下文没上文 /第一个词
    goal=max(candidates(word),key=lambda item:prob_first(item,w3,word,alpha))
    if goal!=word:
        update(goal,word)
    return goal

def main():
    start=time.clock()
    #直接对已经处理过non-word error的语料进行处理
    testdata=readfile.read_file('E:\\!!!!!!!!!!study\\学习\\大四上\\NLP\\NLP intro\\homework\\1\\spelling-correction\\Assignment1\\'+'result6.txt')
    testdata.columns=(['id','senten'])
    result_list=[]
    for i in range(100):
        can=[]
        words=nltk.word_tokenize(testdata['senten'][i])
        for j in range(len(words)):
            p=re.findall(r"[,.?!']",words[j])
            n=re.findall(r"[0-9]+",words[j])
            if(len(p)==0 and len(n)==0):
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
    for i in range(len(result_list)):
        words=result_list[i]
        sen=result_list[i][0].capitalize()
        for j in range(1,len(words)):
            if words[j] in ',.?!':
                sen=sen+words[j]
            else:
                sen=sen+' '+words[j]
        result.append(sen)
    f = open("E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/result_iter.txt",'w')
    for i in range(len(result)):
        f.write(str(i+1)+'\t'+result[i]+'\n') 
    f.close() 
    #结果
    anspath='E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/ans.txt'
    resultpath='E:/!!!!!!!!!!study/学习/大四上/NLP/NLP intro/homework/1/spelling-correction/Assignment1/result_iter.txt'
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