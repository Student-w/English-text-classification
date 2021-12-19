实验围绕用户对服装网站客户的评论数据展开。

1、数据预处理：
  删除空评论、去除标点、提取形容词和动词、去除停用词
  计算文本中感叹号的数量和词语的情感极性作为用户推荐购买意向反映的特征
  
2、将文本转化为TF-IDF矩阵

3、机器学习分类：
  支持向量机（SVC）
  逻辑回归（LR）
  AdaBoost
  随机森林（RF）

4、计算结果：
  RF的F1-score为0.76，其分类性能最优；
  其次是SVC、LR，其F1-score分别为0.72和0.73；
  AdaBoost的分类性能最差，其F1-score为0.71。