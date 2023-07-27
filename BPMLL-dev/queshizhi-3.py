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
from scipy.special import boxcox1p
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