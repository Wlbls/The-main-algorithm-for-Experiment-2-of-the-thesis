#必须导入的包
import pandas as pd
import numpy as np

# 图形可视化库
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# 统计（科学计算库）
from scipy import stats
from scipy.stats import skew, norm
from subprocess import check_output
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# 分类（机器学习常用sklearn库）
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
train = pd.read_csv('Heart-Kidney Yang Deficiency Syndrome_train_data5(8.2).csv')
#导入测试数据集
test = pd.read_csv('Heart-Kidney Yang Deficiency Syndrome_test_data5(8.2).csv')
#训练数据集指标检查
conlumns_show = train.columns
print(conlumns_show)
#打出本和特征的数量
print('The train data size before dropping Id feature is : {} '.format(train.shape))
print('The test data size before dropping Id feature is : {} '.format(test.shape))


#异常值处理（先找出离群点，再按照缺失值处理方法进行；用均值，中位数等数据填充）
#所有特征热力图
corrmat = train.corr() #得到相关系数
f, ax = plt.subplots(figsize=(12, 9)) #一纸多图
sns.heatmap(corrmat, vmax=.8, square=True);

#需要预测的特征证素（以"Heart Blood Stasis Syndrome"列的标签为例子）热力图，哪些特征与它最相关。
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Heart-Kidney Yang Deficiency')['Heart-Kidney Yang Deficiency'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#与Heart Blood Stasis Syndrome高相关的特征的散点图绘制
sns.set()
cols = ['Heart-Kidney Yang Deficiency', 'Diastolic Blood Pressure', 'Systolic Blood Pressure', 'Heart Rate', 'History of Acupuncture Treatment', 'Other Color', 'S-DushuL', 'Total Number of Acupuncture', 'Symptoms similar', 'S-Shenmen']
sns.pairplot(train[cols], size = 2.5)
plt.show();

