import pandas as pd

# 显示所有 行、列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_excel('poke.xlsx', index_col='id')
data = data.drop('Column1', axis=1)    # 删除列
data['种族值'] = data['HP'] + data['攻击'] + data['防御'] + data['速度']
data['新种族值'] = data['HP'] * 1.4 + data['攻击'] * 0.9 + data['防御'] * 0.8 + data['速度'] * 1.5
data['胜负'] = data['新种族值']
# print(data)


HP2 = int(data['HP'].mean())
atack = int(data['攻击'].mean())
defance = int(data['防御'].mean())
space = int(data['速度'].mean())

def function1(i):
    if(i > HP2):
        i = 1
    else:
        i = 0
    return i

def function2(i):
    if(i > atack):
        i = 1
    else:
        i = 0
    return i

def function3(i):
    if(i > defance):
        i = 1
    else:
        i = 0
    return i

def function4(i):
    if(i > space):
        i = 1
    else:
        i = 0
    return i

def function5(i):
    if(i > 320):
        i = 1
    else:
        i = 0
    return i

data['HP'] = data['HP'].apply(function1)
data['攻击'] = data['攻击'].apply(function2)
data['防御'] = data['防御'].apply(function3)
data['速度'] = data['速度'].apply(function4)
data['胜负'] = data['胜负'].apply(function5)
# print(data)

x = data.iloc[:, 1:5].values.astype(int)  # 3 4 5 6列 (自变量)
y = data.iloc[:, 7].values.astype(int)  # 胜负 (因变量)

# print("x: ", x)
# print("y: ", y)


# 划分数据为 训练集与 测试集, 训练集占 70%, 测试集占 30%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=13)

# 生成决策树模型, 并使用训练数据训练该模型
from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')  # 建立决策树模型, 基于信息上
dtc.fit(X_train, Y_train)

# print("X_test: ", X_test)
# 1.自身数据预测
y_pred = dtc.predict(X_test)
# print("自身数据预测 y_pred: ", y_pred)

# 2.自定义数据预测
X_test_dev = [[0, 0, 0, 0],
              [1, 1, 1, 1],
              [1, 0, 0, 1],
              [0, 1, 1, 0]
              ]
# print("X_test: ", X_test_dev)
y_pred_dev = dtc.predict(X_test_dev)
# print("自定义数据预测 y_pred: ", y_pred_dev)

# 模型评估
from sklearn.metrics import classification_report
# 输出 预测准确性
print("score: ", dtc.score(X_test, Y_test))