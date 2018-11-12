# -*- coding: utf-8 -*-
"""
2018实训

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer

from sklearn.linear_model.logistic import LogisticRegression
from sklearn import tree  

import tkinter as tk
import tkinter.messagebox
import pickle 

import warnings
warnings.filterwarnings('ignore') 

#创建字典函数
def creatDictKV(keys, vals):
    lookup = {}
    if len(keys) == len(vals):
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i]
            lookup[key] = val
    return lookup

#AUC函数
def computeAUC(y_true,y_score):
    auc = roc_auc_score(y_true,y_score)
    print("auc=",auc)
    return auc

def analysis():
    #1，加载数据（训练和测试）和预处理数据
    colnames = ['ID','label','loan_amnt','loan_mon','int_rate',
    'installment','grade','home_ownership','annual_inc','verification_status','issue_d',
    'loan_status','dti','earliest_cr_line','inq_last_6mths','open_acc','pub_rec',
    'revol_bal','revol_util','total_acc']
    col_nas = ['', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA','NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA','NA', 'NA', 'NA']
    col_na_values = creatDictKV(colnames, col_nas)
    dftrain = pd.read_csv("data\lendtrain1.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    #print(dftrain)
    train_id = [int(x) for x in dftrain.pop("ID")]
    y_train = np.asarray([int(x)for x in dftrain.pop("label")])
    x_train = dftrain.as_matrix()

    dftest = pd.read_csv("data\lendtest.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    test_id = [int(x) for x in dftest.pop("ID")]
    y_test = np.asarray(dftest.pop("label"))
    x_test = dftest.as_matrix()
    print(y_train)

    #2，使用StratifiedShuffleSplit将训练数据分解为training_new和test_new（用于验证模型）
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train_new, x_test_new = x_train[train_index], x_train[test_index]
        y_train_new, y_test_new = y_train[train_index], y_train[test_index]

    y_train = y_train_new
    x_train = x_train_new

    #3，使用Imputer将NaN替换为平均值
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    x_test_new = imp.transform(x_test_new)
    x_test = imp.transform(x_test)

    #4，使用training_new数据建立RF模型：
    #a.设置rf的参数class_weight="balanced"为"balanced_subsample"
    #n_samples / (n_classes * np.bincount(y))
    rf = RandomForestClassifier(n_estimators=100,
                                oob_score= True,
                                min_samples_split=2,
                                min_samples_leaf=50,
                                n_jobs=-1,
                                class_weight='balanced_subsample',
                                bootstrap=True)
    
    #模型比较
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    predicted_probs_train = lr.predict_proba(x_train)
    predicted_probs_train = [x[1] for  x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)
    
    predicted_probs_test_new = lr.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)
    
    model = tree.DecisionTreeClassifier()    
    model.fit(x_train, y_train)
    predicted_probs_train = model.predict_proba(x_train)
    predicted_probs_train = [x[1] for  x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)
    
    predicted_probs_test_new = lr.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)
    
    #输出特征重要性评估
    rf.fit(x_train, y_train)
    print("importance:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), dftrain.columns),reverse=True))

    #b.使用具有CrossValidation的网格搜索执行参数调整
    param_grid = {"max_features": [2, 3, 4], "min_samples_leaf":[50]}
    grid_search = GridSearchCV(rf, cv=10, scoring='roc_auc', param_grid=param_grid, iid=False)

    #c.输出最佳模型并对测试数据进行预测
    #使用最优参数和training_new数据构建模型
    grid_search.fit(x_train, y_train)
    print("the best parameter:", grid_search.best_params_)
    print("the best score:", grid_search.best_score_)

    #使用训练的模型来预测train_new数据
    predicted_probs_train = grid_search.predict_proba(x_train)
    predicted_probs_train = [x[1] for  x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)

    #使用训练的模型来预测test_new数据（validataion data）
    predicted_probs_test_new = grid_search.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)

    #使用该模型预测test data
    predicted_probs_test = grid_search.predict_proba(x_test)
    predicted_probs_test = ["%.9f" % x[1] for x in predicted_probs_test]
    submission = pd.DataFrame({'Id':test_id, 'Probability':predicted_probs_test})
    submission.to_csv("rf_benchmark.csv", index=False)

#ui界面
def ui():
    window = tk.Tk()
    window.title('违约预测')


    # 用户信息
    tk.Label(window, text='用户ID:').grid(row=0,column=0)
    tk.Label(window, text='年龄:').grid(row=1,column=0)
    tk.Label(window, text='性别:').grid(row=2,column=0)
    tk.Label(window, text='教育程度:').grid(row=3,column=0)
    tk.Label(window, text='信用卡记录:').grid(row=4,column=0)
    tk.Label(window, text='收入情况:').grid(row=5,column=0)
    tk.Label(window, text='成功借款次数:').grid(row=6,column=0)
    tk.Label(window, text='正常还清次数').grid(row=0,column=2)
    tk.Label(window, text='逾期次数15天内').grid(row=1,column=2)
    tk.Label(window, text='逾期15天以上次数').grid(row=2,column=2)
    tk.Label(window, text='累计借款金额:').grid(row=3,column=2)
    tk.Label(window, text='待还金额:').grid(row=4,column=2)
    tk.Label(window, text='单笔最高借款金额').grid(row=5,column=2)
    tk.Label(window, text='历史最高负债').grid(row=6,column=2)
 
    a = []
    var_usr_id= tk.FloatVar()
    entry_usr_id = tk.Entry(window, textvariable=var_usr_id).grid(row=0,column=1)
    a.append(entry_usr_id)
    var_usr_age= tk.FloatVar()
    entry_usr_age = tk.Entry(window, textvariable=var_usr_age).grid(row=1,column=1)
    a.append(entry_usr_age)
    var_usr_sex= tk.FloatVar()
    entry_usr_sex = tk.Entry(window, textvariable=var_usr_sex).grid(row=2,column=1)
    a.append(entry_usr_sex)
    var_usr_edu= tk.FloatVar()
    entry_usr_edu = tk.Entry(window, textvariable=var_usr_edu).grid(row=3,column=1)
    a.append(entry_usr_edu)
    var_usr_credit= tk.FloatVar()
    entry_usr_credit = tk.Entry(window, textvariable=var_usr_credit).grid(row=4,column=1)
    a.append(entry_usr_credit)
    var_usr_income= tk.FloatVar()
    entry_usr_income = tk.Entry(window, textvariable=var_usr_income).grid(row=5,column=1)
    a.append(entry_usr_income)
    var_usr_secceedLendTime= tk.FloatVar()
    entry_usr_secceedLendTime= tk.Entry(window, textvariable=var_usr_secceedLendTime).grid(row=6,column=1)
    a.append(entry_usr_secceedLendTime)
    var_usr_succeedPayTime= tk.FloatVar()
    entry_usr_succeedPayTime = tk.Entry(window, textvariable=var_usr_succeedPayTime).grid(row=0,column=3)
    a.append(entry_usr_succeedPayTime)
    var_usr_overDueShort= tk.FloatVar()
    entry_usr_overDueShort = tk.Entry(window, textvariable=var_usr_overDueShort).grid(row=1,column=3)
    a.append(entry_usr_overDueShort)
    var_usr_overDueLong= tk.FloatVar()
    entry_usr_overDueLong= tk.Entry(window, textvariable=var_usr_overDueLong).grid(row=2,column=3)
    a.append(entry_usr_overDueLong)
    var_usr_totalPay= tk.FloatVar()
    entry_usr_totalPay = tk.Entry(window, textvariable=var_usr_totalPay).grid(row=3,column=3)
    a.append(entry_usr_totalPay)
    var_usr_unpaid= tk.FloatVar()
    entry_usr_unpaid = tk.Entry(window, textvariable=var_usr_unpaid).grid(row=4,column=3)
    a.append(entry_usr_unpaid)
    var_usr_onetimeMaxlend= tk.FloatVar()
    entry_usr_onetimeMaxlend= tk.Entry(window, textvariable=var_usr_onetimeMaxlend).grid(row=5,column=3)
    a.append(entry_usr_onetimeMaxlend)
    var_usr_hisMaxLend= tk.FloatVar()
    entry_usr_hisMaxLend = tk.Entry(window, textvariable=var_usr_hisMaxLend).grid(row=6,column=3)
    a.append(entry_usr_hisMaxLend)

    var = tk.StringVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    tk.Label(window, textvariable=var, fg='red', font=('consolas', 12), width=30, height=2).grid(row=10,column=3)

    at = type(a)
    # 定义一个函数功能，供点击Button按键时调用，调用命令参数command=函数名
    on_hit = False
    def hit_me():
        global on_hit
        if on_hit == False:
            on_hit = True
            var.set(at)
        else:
            on_hit = False
            var.set('')
 
    # 在窗口界面设置放置Button按键
    b = tk.Button(window, text='点我获取违约可能', font=('consolas', 12), command=hit_me).grid(row=10,column=0)

    window.mainloop()
    
if __name__ == "__main__":
    analysis()
    ui()
