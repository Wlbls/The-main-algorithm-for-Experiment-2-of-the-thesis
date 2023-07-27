#必须导入的数据处理的包
import pandas as pd
import numpy as np

# 图形
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()#颜色风格
sns.set_style('darkgrid')#绘制的图形风格为darkgrid

# 统计
from scipy import stats
from scipy.stats import skew, norm#skew为计算偏度或峰度，norm为计算正态分布相关
from subprocess import check_output
from scipy.special import boxcox1p#scipy的special模块包含了大量函数库
from scipy.stats import boxcox_normmax


# 分类
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# 模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor

#导入训练数据集
train = pd.read_csv('train_data5(8.2).csv')
#导入测试数据集
test = pd.read_csv('test_data5(8.2).csv')
#分离特征与标签,此处的train,test均指导入的数据集
train_labels = train.loc[:,('Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency')].reset_index(drop=True)
train_features = train.drop(labels=['Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency'], axis=1)
test_labels = test.loc[:,('Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency')].reset_index(drop=True)
test_features = test.drop(labels=['Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency'], axis=1)
#print(train_labels)
#print(train_features)
#print(test_labels)
#print(test_features)
#把test和train数据放在同一个数据结构里
features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape

#把标签的test和train数据放在同一个数据结构里
labels = pd.concat([train_labels, test_labels]).reset_index(drop=True)
print("标签的形状:", labels.shape)

#缺失值
#缺失值插补之用0代替(本次未涉及）

#用前一个的数据填充
features['Hypertension'] = features['Hypertension'].fillna(method='ffill')
features['Diabetes'] = features['Diabetes'].fillna(method='ffill')
features['History of Coronary Artery Surgery'] = features['History of Coronary Artery Surgery'].fillna(method='ffill')
features['History of Coronary Angiography'] = features['History of Coronary Angiography'].fillna(method='ffill')
features['Referred Pain'] = features['Referred Pain'].fillna(method='ffill')
features['Symptoms similar'] = features['Symptoms similar'].fillna(method='ffill')
features['Distraught'] = features['Distraught'].fillna(method='ffill')
features['Palpitation'] = features['Palpitation'].fillna(method='ffill')
features['Tightness in Breathing'] = features['Tightness in Breathing'].fillna(method='ffill')
features['Spontaneous remission'] = features['Spontaneous remission'].fillna(method='ffill')
features['Nitroglycerin'] = features['Nitroglycerin'].fillna(method='ffill')
features['Family History'] = features['Family History'].fillna(method='ffill')
features['History of Western Medicine Treatment'] = features['History of Western Medicine Treatment'].fillna(method='ffill')
features['History of TCM Treatment'] = features['History of TCM Treatment'].fillna(method='ffill')
features['History of Acupuncture Treatment'] = features['History of Acupuncture Treatment'].fillna(method='ffill')
features['History of Health Care Drug Therapy'] = features['History of Health Care Drug Therapy'].fillna(method='ffill')
features['Treatment of Other Diseases'] = features['Treatment of Other Diseases'].fillna(method='ffill')
features['Grade of Angina Severity'] = features['Grade of Angina Severity'].fillna(method='ffill')
features['Dents'] = features['Dents'].fillna(method='ffill')
features['Hump'] = features['Hump'].fillna(method='ffill')
features['Speckle'] = features['Speckle'].fillna(method='ffill')
features['Papule'] = features['Papule'].fillna(method='ffill')
features['Other Form'] = features['Other Form'].fillna(method='ffill')
features['Red and Swollen'] = features['Red and Swollen'].fillna(method='ffill')
features['Hyperpigmentation'] = features['Hyperpigmentation'].fillna(method='ffill')
features['Other Color'] = features['Other Color'].fillna(method='ffill')
features['Pain'] = features['Pain'].fillna(method='ffill')
features['Delightful'] = features['Delightful'].fillna(method='ffill')
features['Aching and Expand'] = features['Aching and Expand'].fillna(method='ffill')
features['Strip'] = features['Strip'].fillna(method='ffill')
features['Skin temperature arisen'] = features['Skin temperature arisen'].fillna(method='ffill')
features['Other performance of palpation'] = features['Other performance of palpation'].fillna(method='ffill')
#缺失值插补之用平均数替代
#features['Age'] = features.fillna(value=features['Age'].mean())
features = features.fillna(value={'Age':features['Age'].mean(),'Height':features['Height'].mean(),'Weight':features['Weight'].mean(),'Duration of Hypertension (in years)':features['Duration of Hypertension (in years)'].mean(),'Duration of Ddiabetes (years)':features['Duration of Ddiabetes (years)'].mean(),'Body Temperature':features['Body Temperature'].mean()})
features = features.fillna(value = {'Diastolic Blood Pressure':features['Diastolic Blood Pressure'].mean(),'Breathing':features['Breathing'].mean(),'Heart Rate':features['Heart Rate'].mean()})
features = features.fillna(value = {'Duration of Angina Pectoris (in years)':features['Duration of Angina Pectoris (in years)'].mean(),'Number of Angina Attacks (/week)':features['Number of Angina Attacks (/week)'].mean(),'Duration of Angina (/minute)':features['Duration of Angina (/minute)'].mean()})
features = features.fillna(value = {'Total Number of Acupuncture':features['Total Number of Acupuncture'].mean(),'Number of Angina Attacks in Last 4 Weeks (/4 weeks)':features['Number of Angina Attacks in Last 4 Weeks (/4 weeks)'].mean(),'Dose of Nitroglycerin for Last 4 weeks (/4 weeks)':features['Dose of Nitroglycerin for Last 4 weeks (/4 weeks)'].mean(),'VAS Score':features['VAS Score'].mean()})
features = features.fillna(value={'H-Jiquan':features['H-Jiquan'].mean(),'L-Jiquan':features['L-Jiquan'].mean()})
features = features.fillna(value={'ATP-Jiquan':features['ATP-Jiquan'].mean(),'H-Shenmen':features['H-Shenmen'].mean()})
features = features.fillna(value={'L-Shenmen':features['L-Shenmen'].mean(),'ATP-Shenmen':features['ATP-Shenmen'].mean(),'ATP-DushuR':features['ATP-DushuR'].mean(), 'ATD-Qvze':features['ATD-Qvze'].mean()})
#省略后续，直接在Excel中进行计算然后填充，以简略代码
#缺失值插补之剩余所有的NA全部用None代替
#'Painful Region','The Fixed Time', 'The Similar Symptoms', 'Inductive Factors', 'Sensitive form of Shenmen', 'Sensitive form of Yinxi', 'Sensitive form of Shaohai', 'Sensitive form of Neiguan', 'Sensitive form of Ximen', 'Sensitive form of Qvze', 'Sensitive form of Danzhong', 'Sensitive form of Jvque', 'Sensitive form of JueyinshuL', 'Sensitive form of XinshuL', 'Sensitive form of JueyinshuR', 'Sensitive form of XinshuR', 'Sensitive form of DushuL', 'Sensitive form of DushuR'
for col in ('Painful Region','The Fixed Time', 'The Similar Symptoms', 'Inductive Factors', 'SF-Shenmen', 'SF-Yinxi', 'SF-Shaohai', 'SF-Neiguan', 'SF-Ximen', 'SF-Qvze', 'SF-Danzhong', 'SF-Jvque', 'SF-JueyinshuL', 'SF-XinshuL', 'SF-JueyinshuR', 'SF-XinshuR', 'SF-DushuL', 'SF-DushuR'):
    features[col] = features[col].fillna('None')
print("===============填充完毕=================")
#检查是否还有缺失
#print(features.notnull().nunique())


#一些特征其被表示成数值特征缺乏意义，例如年份、类别等，这里将其转换为字符串，即类别型变量。[我的判断方法：即除了0、1之外能用类别字符串替代当前数值的数据要进行修改]
features['Sex'] = features['Sex'].apply(str)
#数据转换（规范化处理）
#先找出数值特征
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes:
        numeric.append(i)
#为所有特征做箱型图（离群点越多，越不符合正态分布）
sns.set_style('white')
f, ax = plt.subplots(figsize=(26, 24))#设置了图像大小
ax.set_xscale('log')
ax = sns.boxplot(data=features[numeric] , orient='h', palette='Set1')
ax.xaxis.grid(False)
ax.set(ylabel='Feature names')
ax.set(xlabel='Numeric values')
ax.set(title='Numeric Distribution of Features')
sns.despine(trim=True, left=True)
plt.show()
#找到倾斜的数值特征
skew_features = features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
print('There are {} numerical features with Skew > 0.5 :'.format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head
#倾斜特征正规化
for i in skew_index:
    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))
#确认是否处理完所有倾斜特征（离群点越少，越符合正态分布）
sns.set_style('white')
f, ax = plt.subplots(figsize=(26, 24))
ax.set_xscale('log')
ax = sns.boxplot(data=features[skew_index] , orient='h', palette='Set1')#设置为竖直排布，并且指定了色调
ax.xaxis.grid(False)
ax.set(ylabel='Feature names')
ax.set(xlabel="Numeric values")
ax.set(title='Numeric Distribution of Features')
sns.despine(trim=True, left=True)#sns.despine()函数默认移除了上部和右侧的轴
plt.show()

#编码分类特征
#机器学习模型需要的数据是数字型的，只有数字类型才能进行计算。因此，对于分类等特殊的特征值，都需要对其进行相应的编码（本次不涉及）

#增加特征
#计算出5项SAQ的标准积分
#西雅图心绞痛量表第一项的标准积分
features['SAQPL'] =(features['SAQ Scale Score 1'] + features['SAQ Scale Score 2'] +features['SAQ Scale Score 3']+features['SAQ Scale Score 4']+features['SAQ Scale Score 5']+features['SAQ Scale Score 6']+features['SAQ Scale Score 7']+features['SAQ Scale Score 8']+features['SAQ Scale Score 9']-9)/45*100
features['SAQAS'] =(features['SAQ Scale Score 10']-1)/4*100
features['SAQAF'] =(features['SAQ Scale Score 11'] + features['SAQ Scale Score 12']-1)/4*100
features['SAQTS'] =(features['SAQ Scale Score 13'] + features['SAQ Scale Score 14'] +features['SAQ Scale Score 15']+features['SAQ Scale Score 16']-4)/16*100
features['SAQDS'] =(features['SAQ Scale Score 17'] + features['SAQ Scale Score 18'] +features['SAQ Scale Score 19']-3)/12*100

print(features.shape)

#字符型特征的独热编码
#查看当前的数据情况是否还存在字符型特征变量没有处理
features.info()
#将其转化为数值型的特征，pandas的get_dummies函数独热编码是默认对所有字符串类型的列进行独热编码。
features = pd.get_dummies(features)

#再看：
features.info()
print("===============至此，所有字符型特征变量都编码成了数值型特征=================")



#使用套索回归模型（Lasso）的系数来刻画特征的重要性图，以此来选择出利于模型训练的关键特征，从而达到特征降维的目的
#首先，将数据拆分回训练数据和测试数据：
y_train = labels.iloc[:len(train_labels), :]
y_test = labels.iloc[len(train_labels):, :]
x_train = features.iloc[:len(train_labels), :]
x_test = features.iloc[len(train_labels):, :]
print(x_train.shape, y_train.shape, x_test.shape)
#然后，需要对特征进行归一化：
scaler = RobustScaler()#RobustScaler 函数使用对异常值鲁棒的统计信息来缩放特征
x_train = scaler.fit(x_train).transform(x_train)  #训练样本特征归一化
x_test = scaler.transform(x_test)               #测试集样本特征归一化

#把dataframe与ndarray互相转换
# y_train = y_train.values
# y_test = y_test.values
# x_train = pd.DataFrame(x_train)
# x_test = pd.DataFrame(x_test)


# # 如果原始文件是.csv格式，则保存文件如下（下面是以自己的数据集为例）
# # 先用pandas读入csv
# data1 = pd.read_csv('train_data5(8.2).csv')
# data2 = pd.read_csv('test_data5(8.2).csv')
# # 再使用numpy保存为npy
# np.save("./dataset/zhengsu/train_data5(8.2).npy", data1)
# np.save("./dataset/zhengsu/test_data5(8.2).npy", data2)
x_train = x_train.astype(np.float64)
y_train = y_train.astype(np.float64)
x_test = x_test.astype(np.int64)
y_test = y_test.astype(np.int64)
dataset_name = 'zhengxing'
np.save('./dataset/' + dataset_name + '/' + 'x_train.npy', x_train)
np.save('./dataset/' + dataset_name + '/' + 'y_train.npy', y_train)
np.save('./dataset/' + dataset_name + '/' + 'x_test.npy', x_test)
np.save('./dataset/' + dataset_name + '/' + 'y_test.npy', y_test)





#特征的选择--基于特征重要性图来选择:
from sklearn.linear_model import Lasso
lasso_model=Lasso(alpha=0.001)
lasso_model.fit(x_train,y_train)
temp = lasso_model.coef_
#选出temp中非全部为0的行，放入下面循环中
#写一个循环分别画出每个标签的图
li_1 = [0,1,2,3,4,5]
for i in li_1[:]:
    u1 = lasso_model.coef_[i]
    ## 索引和重要性做成dataframe形式
    FI_lasso = pd.DataFrame({"Feature Importance":u1},index=features.columns)
    # 由高到低进行排序
    importance_df = FI_lasso.sort_values("Feature Importance",ascending=False).round(5)
    print(importance_df)
    #获取重要程度大于0的系数指标
    FI_lasso[FI_lasso['Feature Importance'] >0 ].sort_values('Feature Importance').plot(kind='barh',figsize=(12,40), color='b')
    plt.xticks(rotation=90)
    plt.show()
    ##画图显示
    FI_index = FI_lasso.index
    FI_val = FI_lasso['Feature Importance'].values
    FI_lasso = pd.DataFrame(FI_val, columns = ['Feature Importance'], index = FI_index)
    print(FI_lasso.shape)
    #据此选出最后的特征
    choose_cols = FI_lasso.index.tolist()
    choose_data = features[choose_cols].copy()
    #然后手动修改最后的特征，即直接在数据集修改，不用代码修改了
#主成分分析
pca_model = PCA(n_components=199)
x_train = pca_model.fit_transform(x_train)
x_test = pca_model.transform(x_test)

#@Time      :2018/11/9 11:02
#@Author    :zhounan


# import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from skmultilearn.dataset import load_from_arff



# （1）np.save(file, arr, allow_pickle=True, fix_imports=True)
# 解释：Save an array to a binary file in NumPy .npy format。以“.npy”格式将数组保存到二进制文件中。
# 参数：
# file 要保存的文件名称，需指定文件保存路径，如果未设置，保存到默认路径。其文件拓展名为.npy
# arr 为需要保存的数组，也即把数组arr保存至名称为file的文件中。
# （2）np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding=‘ASCII’)
# 解释：Load arrays or pickled objects from .npy, .npz or pickled files.


#如果原始文件是.npy格式，则保存文件如下（下面是以鸢尾花数据集为例）
# def pro_iris_data():
#     x = np.load('./dataset/iris/x_train.npy')
#     y = np.load('./dataset/iris/y_train.npy')
#     x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=0)
#     np.save('./dataset/iris/x_train.npy', x_train)
#     np.save('./dataset/iris/y_train.npy', y_train.astype('int64'))
#
#     np.save('./dataset/iris/x_test.npy', x_test)
#     np.save('./dataset/iris/y_test.npy', y_test.astype('int64'))


#如果原始为arr格式，则保存数据文件如下：
# def arff2npy(dataset_name, label_count, split=True):
#
#     if split:
#         train_file_path = './dataset/original_dataset/' + dataset_name + '/' + dataset_name + '-train.arff'
#         test_file_path = './dataset/original_dataset/' + dataset_name + '/' + dataset_name + '-test.arff'
#
#         x_train, y_train = load_from_arff(train_file_path,
#                                           label_count=label_count,
#                                           input_feature_type='float',
#                                           label_location='end',
#                                           load_sparse=False)
#         x_test, y_test = load_from_arff(test_file_path,
#                                           label_count=label_count,
#                                           input_feature_type='int',
#                                           label_location='end',
#                                           load_sparse=False)
#     else:
#         file_path = './dataset/original_dataset/' + dataset_name + '/' + dataset_name + '.arff'
#         x, y = load_from_arff(file_path,
#                               label_count=label_count,
#                               input_feature_type='int',
#                               label_location='end',
#                               load_sparse=True)
#
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)
#
#     # x_train = x_train.astype(np.float64)
#     # y_train = y_train.astype(np.int64)
#     # x_test = x_test.astype(np.float64)
#     # y_test = y_test.astype(np.int64)
#
#     np.save('./dataset/'+ dataset_name +'/x_train.npy', x_train.toarray())
#     np.save('./dataset/'+ dataset_name +'/y_train.npy', y_train.toarray())
#     np.save('./dataset/'+ dataset_name +'/x_test.npy', x_test.toarray())
#     np.save('./dataset/'+ dataset_name +'/y_test.npy', y_test.toarray())

def loadnpy(dataset_name):
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# #用yeast来跑的话如下：
# if __name__ == '__main__':
#     dataset_name = 'yeast'
#     arff2npy(dataset_name, label_count=14, split=True)
#     loadnpy(dataset_name)

#用delicious来跑的话如下：
# if __name__ == '__main__':
#     dataset_name = 'delicious'
#     arff2npy(dataset_name, label_count=500, split=True)
#     loadnpy(dataset_name)

#用自己数据集来跑的话如下：
# if __name__ == '__main__':
#     dataset_name = 'Heart Blood Stasis Syndrome'
#     # arff2npy(dataset_name, label_count=21, split=True)
#     loadnpy(dataset_name)