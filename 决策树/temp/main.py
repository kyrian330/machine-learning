"""
* @描述: 决策树例子
* @作者: 草莓夹心糖
* @创建时间: 2022/11/8 20:23
"""

import pandas as pd
import sys
try:
    data = pd.read_csv('clf.csv')
except:
    print("请确保数据集存在, 且读取路径无误")
    sys.exit()
# data = pd.read_csv('classify.csv')
# print(data)

# 取出特征数据
X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
# 取出类别数据
Y = data['y'].values
# print(X)
# print(Y)

from sklearn.model_selection import train_test_split
# 拆分数据
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn import tree
# 建模
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train, Y_train)
# score = clf.score(X_test, Y_test)   # 返回预测准确度
# print(score)


# 绘图
feature_name = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
try:
    import graphviz
except:
    print("请正确安装graphviz")
    sys.exit()
dot_data = tree.export_graphviz(clf
                               ,out_file=None
                               ,feature_names=feature_name
                               ,class_names=['0', '1', '2']
                               ,filled=True
                               ,rounded=True
                               )
graph = graphviz.Source(dot_data)
# print(graph)
graph.render('决策树可视化')


# 保存模型
import joblib
import os
if not os.path.exists('models'):
    os.makedirs('models')    # 如果 models文件夹不存在就创建
    print("模型将保存到 models文件夹")
joblib.dump(clf, 'models/clf.pkl')