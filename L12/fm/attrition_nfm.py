# 员工离职预测
import pandas as pd
import numpy as np

# 数据加载
train = pd.read_csv('./attrition/train.csv')
test = test1 = pd.read_csv('./attrition/test.csv')
#print(train['Attrition'].value_counts())

# 设置标记位
test['Attrition']=-1
test = test[train.columns]
data = pd.concat([train, test])
# 处理Attrition字段
data['Attrition']=data['Attrition'].map(lambda x:1 if x=='Yes' else -1 if x==-1 else 0)

# 分类特征
cate = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
# 使用factorize将离散值对应为 index值
for feature in cate:
    data[feature] = pd.factorize(data[feature])[0]
#data.to_csv('temp.csv')
#print(data)

# 去掉没用的列 员工号码，标准工时（=80）

data = data.drop(['user_id', 'EmployeeNumber', 'StandardHours'], axis=1)
# 训练集测试集分离, 如果Attrition=-1 说明是测试集
train, test = data[data['Attrition']!=-1], data[data['Attrition']==-1]
#train = train.drop('Attrition', axis=1)
#print(train)
#print(len(train[train['Attrition']==1]),len(train[train['Attrition']==0]))

from sklearn.model_selection import train_test_split
# 划分训练集验证集
#X_train, X_test, y_train, y_test = train_test_split(train.drop('Attrition', axis=1), train['Attrition'], test_size=0.2, random_state=111)

from deepctr.models import DeepFM, NFM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.inputs import  SparseFeat, DenseFeat,get_feature_names

sparse_features = cate
# 除了分类特征以外，其余都是稠密类型
dense_features = list(set([i if i not in cate else '' for i in train.drop('Attrition', axis=1).columns]))
dense_features.remove('')
# 处理缺失值
train[sparse_features] = train[sparse_features].fillna('-1', )
train[dense_features] = train[dense_features].fillna(0, )
# 对离散特征进行标签编码
target = ['Attrition']
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
#print(data)

# 对数据进行0-1规划反
mms = MinMaxScaler(feature_range=(0, 1))
train[dense_features] = mms.fit_transform(train[dense_features])
test[dense_features] = mms.fit_transform(test[dense_features])
# 处理定长离散特征
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
print(fixlen_feature_columns)

# 得到所有特证名
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train_model_input = {name:train[name] for name in feature_names}
model = NFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=50, verbose=2, validation_split=0.2, )

# 对测试集进行预测
test_model_input = {name:test[name] for name in feature_names}
#print(test_model_input)
predict = model.predict(test_model_input, batch_size=256)
## 转化为二分类输出
test1['Attrition'] = predict
test1['Attrition']=test1['Attrition'].map(lambda x:1 if x>=0.5 else 0)
# 使用user_id作为索引
test1.set_index(["user_id"], inplace=True)
test1[['Attrition']].to_csv('submit_nfm.csv')
