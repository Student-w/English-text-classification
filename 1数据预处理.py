# -*- coding: utf-8 -*-
"""
@author: Student

数据预处理部分包含：

1、数据清洗：空评论、包含数字、标点、与推荐倾向无关词汇

2、文本信息抽取：感叹号的数量、词语的情感极性、提取文本评论中的动词和形容词

3、处理后的数据存为'text_df.csv'文档，方便下次直接读取训练模型

"""

### 数据预处理
import pandas as pd
import numpy as np

text_df = pd.read_csv("./服装网站评论数据.csv", index_col=0)
print(text_df.shape)
text_df.head()#显示数据前5行

## 将'Title', 'Review Text'合并，存为新指标Review，并将原指标删除
text_df['Review'] = text_df['Title'] + ' ' + text_df['Review Text']#合并，生成新的Review列
text_df = text_df.drop(labels=['Title','Review Text'] , axis=1)#删去原来的两列
print("My data's shape is:", text_df.shape)
text_df.head()

## 删除空评论
text_df.Review.isna().sum()#空评论的数量
text_df = text_df[~text_df.Review.isna()]#保留非空评论
text_df = text_df.rename(columns={"Recommended IND": "Recommended"})#对Recommended IND列重命名
print("My data's shape is:", text_df.shape)
text_df.head()

## 统计评论中感叹号的个数,加入到数据集中作为文本分类的特征
def count_exclamation_mark(string_text):
    count = 0
    for char in string_text:
        if char == '!':
            count += 1
    return count
text_df['count_exc'] = text_df['Review'].apply(count_exclamation_mark)
text_df['count_exc'].describe()
text_df.head(5)

## 删除标点符号
import string
string.punctuation#所有的标点符号
#编写函数，仅保留标点符号集中没有的字符
def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str
text_df['Review'] = text_df['Review'].apply(punctuation_removal)
text_df['Review'].head()

## 去停用词
import nltk
from nltk.corpus import stopwords
# 加载停用词
#nltk.download('stopwords')
stop = stopwords.words('english')
stop.append("i'm")
# 编写函数：移除停用词中的标点
stop_words = []
for item in stop: 
    new_item = punctuation_removal(item)
    stop_words.append(new_item) 
print(stop_words[::12])
# 服装类中性停用词
clothes_list =['dress', 'top','sweater','shirt',
               'skirt','material', 'white', 'black',
              'jeans', 'fabric', 'color','order', 'wear']
# 去停用词
from nltk.tokenize import word_tokenize
def stopwords_removal(str):
    str = word_tokenize(str)
    return [word.lower() for word in str 
            if word.lower() not in stop_words and word.lower() not in clothes_list]

text_df['Review'] = text_df['Review'].apply(stopwords_removal)
text_df['Review'].head()

## 去除文本中包含的数字
import re
#re.search()方法扫描整个字符串，并返回第一个成功的匹配。如果匹配失败，则返回None '\d'匹配任意数字
def numbers_removal(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\d', i):
            list_text_new.append(i)
    return ' '.join(list_text_new)#注意return []返回的是列表，return ''.join()返回的是字符串

text_df['Review'] = text_df['Review'].apply(numbers_removal)
text_df['Review'].head()

## 评论文本的情感极性，作为判断用户是否推荐的一个特征
'''
Polarity is the emotion expressed in the sentence. It can be positive, neagtive and neutral.
The polarity score is a float within the range [-1.0, 1.0]
TextBlob用来执行很多自然语言处理的任务，比如:词性标注，名词性成分提取，情感分析，文本翻译，等等
SnowNLP是一个python写的类库,可以方便的处理中文文本内容
'''
from textblob import TextBlob
text_df['Polarity'] = text_df['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)
text_df.head(5)

## 提取评论中的形容词和动词：大多数形容词和动词反映了评论的极性
'''NLTK是Python很强大的第三方库，可以很方便的完成很多自然语言处理（NLP）的任务，
包括分词、词性标注、命名实体识别（NER）及句法分析'''
#定义函数，使用nltk包提取评论中的形容词和动词
def adj_collector(review_string):
    new_string=[]
    review_string = word_tokenize(review_string)#分词
    tup_word = nltk.pos_tag(review_string)#对单词的词性进行标记
    for tup in tup_word:
        if 'VB' in tup[1] or tup[1]=='JJ':  #Verbs and Adjectives
            new_string.append(tup[0])  
    return ' '.join(new_string)

#nltk.download('averaged_perceptron_tagger')
text_df['Review'] = text_df['Review'].apply(adj_collector)
text_df['Review'].head()

# 将处理后的数据保存为Excel文档
Output_path = './text_df.csv'
text_df.to_csv(Output_path, encoding="utf-8")
