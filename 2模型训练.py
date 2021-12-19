# -*- coding: utf-8 -*-
"""
@author: Student

本部分内容包括：

1、将文本转化为TF-IDF矩阵，切分训练集和测试集

2、训练机器学习模型，输出模型的准确率指标

"""
Input_path = './text_df.csv'
text_prep = pd.read_csv(Input_path, index_col=0)
n_row1 = text_prep.shape[0]
text_prep = text_prep[~text_prep['Review'].isna()]#保留非空评论
print("删除空评论的数量：{}条".format(n_row1-text_prep.shape[0]))

### 一、生成TF-IDF矩阵
## Vectorizing - Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

text_prep['Review'].head()
def text_vectorizing_process(sentence_string):
    return [word for word in sentence_string.split()]#split()：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
bow_transformer = CountVectorizer(text_vectorizing_process)#文本特征提取，词袋法，只考虑词汇在文本中出现的频率，对列表操作
bow_transformer.fit(text_prep['Review'].values.astype('U'))

Reviews = bow_transformer.transform(text_prep['Review'].values.astype('U'))

## TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(Reviews)
messages_tfidf = tfidf_transformer.transform(Reviews)
messages_tfidf.shape
messages_tfidf = messages_tfidf.toarray()#将TF-IDF转化成numpy矩阵
messages_tfidf = pd.DataFrame(messages_tfidf)#进而转换成DataFrame
print(messages_tfidf.shape)
messages_tfidf.head()

'''
### 二、word2vec词向量（memory error暂时不能运行成功）
#seeds = list(range(0, text_prep.shape[0]))
#
#import random
#sample = random.sample(seeds, 5000)
Clean_path = './text_prep.csv'
text_prep = pd.read_csv(Clean_path, index_col=0)
text_prep = text_prep[~text_prep['Review'].isna()]#保留非空评论
#text_prep = text_prep.iloc[sample]

Lis = [lis for lis in text_prep['Review'].str.split()]
    
text_prep['Review'].isna().sum()
## 将序列用<PAD/>填充为统一长度
from keras.preprocessing.sequence import pad_sequences
max_len = max(len(lis) for lis in Lis)
Lis_pad = []
for lis in Lis:
    Lis_pad.append(list(pad_sequences([lis], maxlen=max_len, dtype=object, padding='pre',  value='<PAD/>')[0]))
    
## 训练词向量
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore') 
from gensim.models import Word2Vec

content = Lis_pad
path_wv = './Review.model'
model = Word2Vec(content, sg=0, size=200, window=15, min_count=1, workers=4)
model.save(path_wv)

## 将原来的词转化为词向量
model = Word2Vec.load('./Review.model')
Lis_w2v = []
lis_w2v = []
for lis in Lis_pad:
    for i in lis:
        lis_w2v += list(model.wv[i])
    Lis_w2v.append(lis_w2v)

messages_tfidf2 = pd.DataFrame(Lis_w2v)#转换成DataFrame
'''

### 将TF-IDF矩阵和其他特征融合:删除Review列，此时的数据含'Recommended', 'Review_length','count_exc','Polarity',及TF-IDF矩阵
df_all = pd.merge(text_prep.drop(columns='Review'), messages_tfidf, 
                  left_index=True, right_index=True )
df_all.head()

### 分离数据和标签
X = df_all.drop('Recommended', axis=1)
y = df_all.Recommended

## 划分训练集和测试集
from sklearn.model_selection import train_test_split as split

X_train, X_test, y_train, y_test = split(X, y, test_size=0.3, stratify=y, random_state=111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

## 查看训练样本和测试样本的类别比例，比例差别较大 
'''
在训练模型时，参数class_weight='balanced' “平衡”模式使用y的值来自动调整输入数据中与类频率成反比的权重
'''
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

## MinMaxScaler法对数据标准化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import scipy
#把一个稀疏的np.array压缩.csr_matrix：Compressed Sparse Row marix,而csc_matric：Compressed Sparse Column marix。
X_train_scaled = scipy.sparse.csr_matrix(X_train_scaled)
X_test_scaled = scipy.sparse.csr_matrix(X_test_scaled)
X_train = scipy.sparse.csr_matrix(X_train.values)
X_test = scipy.sparse.csr_matrix(X_test.values)

### 分类模型
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt

def report(y_true, y_pred, labels):
    cm = pd.DataFrame(confusion_matrix(y_true=y_true, y_pred=y_pred), 
                                        index=labels, columns=labels)
    rep = classification_report(y_true=y_true, y_pred=y_pred)
    return (f'Confusion Matrix:\n{cm}\n\nClassification Report:\n{rep}')#f''格式化操作

## 1. SVC
from sklearn.svm import SVC
svc_model = SVC(C=1.0, 
             kernel='linear',
             class_weight='balanced', 
             probability=True,
             random_state=111)
svc_model.fit(X_train_scaled, y_train)

test_predictions = svc_model.predict(X_test_scaled)
print('\n\n*SVC*\n\n',report(y_test, test_predictions, svc_model.classes_ ))

skplt.metrics.plot_roc(y_test, svc_model.predict_proba(X_test_scaled),
                      title='ROC Curves - SVC') 

## 2. Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lr_model = LogisticRegression(class_weight='balanced', 
                              random_state=111, 
                              solver='lbfgs',
                              C=1.0)
gs_lr_model = GridSearchCV(lr_model, 
                           param_grid={'C': [0.01, 0.1, 0.5, 1.0, 5.0]}, 
                           cv=5, 
                           scoring='roc_auc')#自动调参
gs_lr_model.fit(X_train_scaled, y_train)

gs_lr_model.best_params_

test_predictions = gs_lr_model.predict(X_test_scaled)
print('\n\n*Logistic Regression*\n\n', report(y_test, test_predictions, gs_lr_model.classes_ ))

skplt.metrics.plot_roc(y_test, gs_lr_model.predict_proba(X_test_scaled),
                      title='ROC Curves - Logistic Regression') 

## 3. AdaBoost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=555)
#决策树为弱分类学习器
ada_model = AdaBoostClassifier(base_estimator=dt, learning_rate=0.001, n_estimators=1000, random_state=222)
ada_model.fit(X_train ,y_train)

test_predictions = ada_model.predict(X_test)
print('\n\n*AdaBoost*\n\n', report(y_test, test_predictions, ada_model.classes_ ))

skplt.metrics.plot_roc(y_test, ada_model.predict_proba(X_test), 
                       title='ROC Curves - AdaBoost') 

## 4. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=1000, max_depth=5, 
                                  class_weight='balanced', random_state=3)
rf_model.fit(X_train, y_train)

test_predictions = rf_model.predict(X_test)

print('\n\n*Random Forest*\n\n', report(y_test, test_predictions, rf_model.classes_ ))

skplt.metrics.plot_roc(y_test, rf_model.predict_proba(X_test), 
                       title='ROC Curves - Random Forest') 

# * SVC - f1 micro score of 0.72
# * Logistic Regression - f1 micro score of 0.71
# * AdaBoost - f1 micro score of 0.73
# * Random Forest - f1 micro score of 0.76
