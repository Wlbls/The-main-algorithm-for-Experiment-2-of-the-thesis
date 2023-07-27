#必须导入的包
import pandas as pd
import numpy as np

# 图形
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# 统计
from scipy import stats
from scipy.stats import skew, norm
from subprocess import check_output
from scipy.special import boxcox1p #对数据进行box-cox11p转换
from scipy.stats import boxcox_normmax #对数据进行boxcox_normmax转换

# 分类
from sklearn.model_selection import GridSearchCV #GridSearchCV是sklearn中的一个函数，专门调试参数的函数。用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。
from sklearn.model_selection import KFold, cross_val_score #交叉验证
from sklearn.metrics import mean_squared_error #MSE（均方误差），属于回归模型性能评价指标之一
from sklearn.preprocessing import OneHotEncoder #独热编码
from sklearn.preprocessing import LabelEncoder #标签编码
from sklearn.pipeline import make_pipeline #Pipeline可以将许多算法模型串联起来，比如将特征提取、归一化、分类组织在一起形成一个典型的机器学习问题工作流。主要带来两点好处：直接调用fit和predict方法来对pipeline中的所有算法模型进行训练和预测。可以结合grid search对参数进行选择。
from sklearn.preprocessing import scale #sklearn.preprocessing下的数据标准化（scale即用均值和标准差来进行数据标准化、MinMaxScaler）
from sklearn.preprocessing import StandardScaler #对特征进行标准化
from sklearn.preprocessing import RobustScaler #RobustScaler函数使用对异常值鲁棒的统计信息来缩放特征。
from sklearn.decomposition import PCA #PCA即主成分分析（然后降维）， 在sklearn中，与PCA相关的类都在sklearn.decomposition包中

# 模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge #核岭回归模型
from sklearn.pipeline import make_pipeline #建造管道
from sklearn.preprocessing import RobustScaler #RobustScaler 函数使用对异常值鲁棒的统计信息来缩放特征
from sklearn.linear_model import Ridge, RidgeCV #均为岭回归模型
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
#训练数据集指标检查
conlumns_show = train.columns
print(conlumns_show)
#打出样本和特征的数量
print('The train data size before dropping Id feature is : {} '.format(train.shape))
print('The test data size before dropping Id feature is : {} '.format(test.shape))

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
labels.shape


#缺失值
#缺失值插补之用0代替(本次未涉及）

#缺失值插补之用任意两个特征分组后的中位数替代(本次未涉及）
#缺失值插补之用来自它们之前和之后的五个数据值的平均值或求和以估计二分类的值（本次未涉及）

#print(features.notnull().nunique())#这里用来判断数据中是否存在为空，并且那些列存在为空的值

#缺失值插补之用常见值代替,mode方法表示求众数，以最多的那个值进行填充(会自动忽略NA数量），但会将NA当作空值一起填充，所以可以运行完代码后注释掉然后手动填充
#features['Painful Region'] = features['Painful Region'].fillna(features['Painful Region'].mode().iloc[0])
#print(features['Painful Region'])
#features['The Fixed Time'] = features['The Fixed Time'].fillna(features['The Fixed Time'].mode().iloc[0])
#print(features['The Fixed Time'])
#features['The Similar Symptoms'] = features['The Similar Symptoms'].fillna(features['The Similar Symptoms'].mode().iloc[0])
#print(features['The Similar Symptoms'])
#features['Inductive Factors'] = features['Inductive Factors'].fillna(features['Inductive Factors'].mode().iloc[0])
#print(features['Inductive Factors'])
#features['Sensitive form of Danzhong'] = features['Sensitive form of Danzhong'].fillna(features['Sensitive form of Danzhong'].mode().iloc[0])
#print(features['Sensitive form of Danzhong'])
#features['Sensitive form of Shenmen'] = features['Sensitive form of Shenmen'].fillna(features['Sensitive form of Shenmen'].mode().iloc[0])
#print(features['Sensitive form of Shenmen'])
#features['Sensitive form of Yinxi'] = features['Sensitive form of Yinxi'].fillna(features['Sensitive form of Yinxi'].mode().iloc[0])
#print(features['Sensitive form of Yinxi'])
#features['Sensitive form of Shaohai'] = features['Sensitive form of Shaohai'].fillna(features['Sensitive form of Shaohai'].mode().iloc[0])
#print(features['Sensitive form of Shaohai'])
#features['Sensitive form of Neiguan'] = features['Sensitive form of Neiguan'].fillna(features['Sensitive form of Neiguan'].mode().iloc[0])
#print(features['Sensitive form of Neiguan'])
#features['Sensitive form of Ximen'] = features['Sensitive form of Ximen'].fillna(features['Sensitive form of Ximen'].mode().iloc[0])
#print(features['Sensitive form of Ximen'])
#features['Sensitive form of Qvze'] = features['Sensitive form of Qvze'].fillna(features['Sensitive form of Qvze'].mode().iloc[0])
#print(features['Sensitive form of Qvze'])
#features['Sensitive form of Jvque'] = features['Sensitive form of Jvque'].fillna(features['Sensitive form of Jvque'].mode().iloc[0])
#print(features['Sensitive form of Jvque'])
#features['Sensitive form of JueyinshuL'] = features['Sensitive form of JueyinshuL'].fillna(features['Sensitive form of JueyinshuL'].mode().iloc[0])
#print(features['Sensitive form of JueyinshuL'])
#features['Sensitive form of XinshuL'] = features['Sensitive form of XinshuL'].fillna(features['Sensitive form of XinshuL'].mode().iloc[0])
#print(features['Sensitive form of XinshuL'])
#features['Sensitive form of JueyinshuR'] = features['Sensitive form of JueyinshuR'].fillna(features['Sensitive form of JueyinshuR'].mode().iloc[0])
#print(features['Sensitive form of JueyinshuR'])
#features['Sensitive form of XinshuR'] = features['Sensitive form of XinshuR'].fillna(features['Sensitive form of XinshuR'].mode().iloc[0])
#print(features['Sensitive form of XinshuR'])
#features['Sensitive form of DushuL'] = features['Sensitive form of DushuL'].fillna(features['Sensitive form of DushuL'].mode().iloc[0])
#print(features['Sensitive form of DushuL'])
#features['Sensitive form of DushuR'] = features['Sensitive form of DushuR'].fillna(features['Sensitive form of DushuR'].mode().iloc[0])
#print(features['Sensitive form of DushuR'])
#features['Painful Region'] = features['Painful Region'].fillna(features['Painful Region'].mode().iloc[0])
#print(features['Painful Region'])



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

#缺失值插补之剩余的缺失值NA用None代替
#'Painful Region','The Fixed Time', 'The Similar Symptoms', 'Inductive Factors', 'Sensitive form of Shenmen', 'Sensitive form of Yinxi', 'Sensitive form of Shaohai', 'Sensitive form of Neiguan', 'Sensitive form of Ximen', 'Sensitive form of Qvze', 'Sensitive form of Danzhong', 'Sensitive form of Jvque', 'Sensitive form of JueyinshuL', 'Sensitive form of XinshuL', 'Sensitive form of JueyinshuR', 'Sensitive form of XinshuR', 'Sensitive form of DushuL', 'Sensitive form of DushuR'
for col in ('Painful Region','The Fixed Time', 'The Similar Symptoms', 'Inductive Factors', 'SF-Shenmen', 'SF-Yinxi', 'SF-Shaohai', 'SF-Neiguan', 'SF-Ximen', 'SF-Qvze', 'SF-Danzhong', 'SF-Jvque', 'SF-JueyinshuL', 'SF-XinshuL', 'SF-JueyinshuR', 'SF-XinshuR', 'SF-DushuL', 'SF-DushuR'):
    features[col] = features[col].fillna('None')

print("===============填充完毕=================")
#检查是否还有缺失
temp = features.notnull().nunique()
print(temp)


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
