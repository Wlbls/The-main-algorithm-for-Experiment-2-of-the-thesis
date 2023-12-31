#@Time      :2018/10/31 22:23
#@Author    :zhounan
# @FileName: bp_mll_test.py
import numpy as np
import tensorflow as tf
#import matpotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
#from sklearn.externals import joblib
import joblib
import evaluate_model
from evaluate_model import TPR, FPR, F1, PRE, RECALL, sort, hamming_loss, find, findmax, sort, findIndex, avgprec, Coverage, OneError, rloss


def predict(x_test, dataset_name, model_type):
    with tf.Session() as sess:
        saver  = tf.train.import_meta_graph('./tf_model/' + dataset_name + '/' + model_type + '_model.meta')
        saver.restore(sess, './tf_model/' + dataset_name + '/' + model_type + '_model')
        graph = tf.get_default_graph()
        pred = tf.get_collection('pred_network')[0]
        x = graph.get_operation_by_name('input_x').outputs[0]

        pred = sess.run(pred, feed_dict={x: x_test})

        linreg = joblib.load('./sk_model/' + dataset_name + '/' + model_type + '_linear_model.pkl')
        thresholds = linreg.predict(pred)
        y_pred = ((pred.T - thresholds.T) > 0).T

        #translate bool to int
        y_pred = y_pred + 0
        return y_pred, pred


#eliminate some data that have full true labels or full false labels
#移除全1或者全0标签
def eliminate_data(data_x, data_y):
    data_num = data_y.shape[0]
    label_num = data_y.shape[1]
    full_true = np.ones(label_num)
    full_false = np.zeros(label_num)

    i = 0
    while(i < len(data_y)):
        if (data_y[i] == full_true).all() or (data_y[i] == full_false).all():
            data_y = np.delete(data_y, i, axis=0)
            data_x = np.delete(data_x, i, axis=0)
        else:
            i = i + 1

    return data_x, data_y

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')
    x_test, y_test = eliminate_data(x_test, y_test)

    return x_train, y_train, x_test, y_test

def ensemble(pos_pred, neg_pred):
    return pos_pred | neg_pred

if __name__ == '__main__':
    # dataset_names = ['yeast','delicious']
    #dataset_names = ['yeast']
    dataset_names = ['zhengxing']
    dataset_name = dataset_names[0]
    _, _, x_test, y_test = load_data(dataset_name)

    model_type = 'positive'
    pos_pred, pos_output = predict(x_test, dataset_name, model_type)
    print(dataset_name, model_type, 'hammingloss:', evaluate_model.hamming_loss(pos_pred, y_test))

    model_type = 'negtive'
    pred, neg_output = predict(x_test, dataset_name, model_type)
    neg_pred = 1 - pred
    print(dataset_name, model_type, 'hammingloss:', evaluate_model.hamming_loss(neg_pred, y_test))
    # print(dataset_name, 'rankingloss:', evaluate_model.rloss(output, y_test))


#ensemble:集成（融合）
    ensemble_pred = ensemble(pos_pred, neg_pred)
    print(dataset_name, 'ensemble hammingloss:', evaluate_model.hamming_loss(ensemble_pred, y_test))

#将原po在“evaluate_moduel"中定义好的函数拿过来直接用
#1-错误率
    print(dataset_name, 'OneError:', OneError(ensemble_pred, y_test))
#排序损失
    print(dataset_name, 'rloss:', rloss(ensemble_pred, y_test))
#平均精度
    print(dataset_name, 'avgprec:', avgprec(ensemble_pred, y_test))
#覆盖率
    print(dataset_name, 'Coverage:', Coverage(ensemble_pred, y_test))
#F1-score
