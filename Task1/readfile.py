# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 08:29:00 2019

@author: Kitty
"""

import pandas as pd
#读取文件
def read_file(path):
    f=open(path,'r')
    data=f.readlines()
    f.close()
    for i in range(len(data)):
        data[i]=data[i].split('\t')
    data=pd.DataFrame(data)
    return data

def readvoc(path):
    f=open(path,'r')
    data=f.readlines()
    for i in range(len(data)):
        data[i]=data[i].strip()
    f.close()
    return data

def load_non():
    #读入数据
    path='E:\\!!!!!!!!!!study\\学习\\大四上\\NLP\\NLP intro\\homework\\1\\spelling-correction\\Assignment1\\'
    testdata=read_file(path+'testdata.txt')
    testdata.columns=(['id','error num','senten'])    
    ans=read_file(path+'ans.txt')
    ans.columns=(['id','corrected'])
    vocab=readvoc(path+'vocab.txt')
    return testdata,ans,vocab
    
def load_trans():
    #读入p(x|w)计算的频数
    path2='E:\\!!!!!!!!!!study\\学习\\大四上\\NLP\\NLP intro\\homework\\1\\'
    transition=read_file(path2+'transition.txt')
    transition.columns=['char','count']
    transition['x'],transition['w']=transition['char'].str.split('|').str
    del transition['char']
    transition['count']=transition['count'].astype('int')
    return transition


def Trans(str2,str1):
    transition=load_trans()
    if len(transition[(transition['x']==str1) & (transition['w']==str2)]['count'])>0:
        return float(transition[(transition['x']==str1) & (transition['w']==str2)]['count'].iloc[0]/sum(transition[transition['w']==str2]['count']))
    elif len(transition[transition['w']==str2])>0:
        return float(1/sum(transition[transition['w']==str2]['count']))
    else:
        return float(1/sum(transition['count']))   #edit2及以上

def prob_edit1():
    path2='E:\\!!!!!!!!!!study\\学习\\大四上\\NLP\\NLP intro\\homework\\1\\'
    transition=read_file(path2+'transition.txt')
    transition.columns=['char','count']
    transition['x'],transition['w']=transition['char'].str.split('|').str
    del transition['char']
    count=0
    transition['count']=transition['count'].astype('int')
    for i in range(len(transition)):
        if(len(transition['x'][i])==1 and len(transition['w'][i])==1):
            count+=transition['count'][i]   #sub
        elif(len(transition['w'][i])>1):
            if(transition['w'][i][0]==transition['x'][i]):
                count+=transition['count'][i]   #del
            elif(len(transition['x'][i])>1):
                if(transition['w'][i][0]==transition['x'][i][1] and transition['w'][i][1]==transition['x'][i][0]):
                    count+=transition['count'][i]   #trans
        elif(len(transition['x'][i])>1):
            if(transition['w'][i][0]==transition['x'][i]):
                count+=transition['count'][i]   #ins
    p_edit1=count/sum(transition['count'])
    return p_edit1
     
    

            