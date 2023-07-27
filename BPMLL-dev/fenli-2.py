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
#打出本和特征的数量
print('The train data size before dropping Id feature is : {} '.format(train.shape))
print('The test data size before dropping Id feature is : {} '.format(test.shape))

#分离特征与标签,此处的train,test均指导入的数据集
train_labels = train.loc[:,('Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency')].reset_index(drop=True)
train_features = train.drop(labels=['Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency'], axis=1)
test_labels = test.loc[:,('Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency')].reset_index(drop=True)
test_features = test.drop(labels=['Heart Blood Stasis','Cold Congealing in Heart Vessel','Blockade of Phlegm-Turbidity','Heart Qi Deficiency','Heart-Kidney Yin Deficiency','Heart-Kidney Yang Deficiency'], axis=1)
print(train_labels)
print(train_features)
print(test_labels)
print(test_features)

#把test和train数据放在同一个数据结构里
features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape