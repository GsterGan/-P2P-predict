# -*- coding: utf-8 -*-
"""

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
from tkinter import ttk

import warnings
warnings.filterwarnings('ignore') 
#RF模型：
#1.加载数据（训练和测试）和预处理数据
#2.将训练数据分解为training_new和test_new（用于验证模型）
#3.用Imputer处理数据：用Mean代替缺失值
#4.使用training_new数据建立RF模型：
#a 处理不平衡的数据分布
#b 使用带有CrossValidation的网格搜索执行参数调整
#c 输出最佳模型并对测试数据进行预测

#创建字典函数
#input: keys =[]and values=[]
#output: dict{}
def creatDictKV(keys, vals):
    lookup = {}
    if len(keys) == len(vals):
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i]
            lookup[key] = val
    return lookup
#计算AUC函数
# input: y_true =[] and y_score=[]
# output: auc
def computeAUC(y_true,y_score):
    auc = roc_auc_score(y_true,y_score)
    print("auc=",auc)
    return auc

on_hit = False
def hit_me():
    global on_hit
    if on_hit == False:
        on_hit = True
        var.set(a[0][1])
    else:
        on_hit = False
        var.set('')

def main():
    #1，加载数据（训练和测试）和预处理数据
    #将NumberTime30-59，60-89，90中标记的96，98替换为NaN
    #将Age中的0替换为NaN
    colnames = ['ID','label','性别','年龄','文化程度','收入情况','征信记录',
                '成功借款次数','成功还款次数',
                '正常还清次数','逾期（0-15天）还清次数',
                '累计借款金额','待还金额','单笔最高借款金额','历史最高负债']
    col_nas = ['','NA','NA','NA',0,'NA',0,'NA',0,'NA',
               'NA','NA','NA','NA','NA','NA','NA']
    col_na_values = creatDictKV(colnames, col_nas)
    dftrain = pd.read_csv("data\ppdai.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    #print(dftrain)
    train_id = [int(x) for x in dftrain.pop("ID")]
    y_train = np.asarray([int(x)for x in dftrain.pop("label")])
    x_train = dftrain.as_matrix()

    dftest = pd.read_csv("data\pptest.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    test_id = [int(x) for x in dftest.pop("ID")]
    y_test = np.asarray(dftest.pop("label"))
    x_test = dftest.as_matrix()
    #2，使用StratifiedShuffleSplit将训练数据分解为training_new和test_new（用于验证模型）
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.33333,random_state=0)
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
    #computeAUC(y_test_new, predicted_probs_test_new)
    
    #输出特征重要性评估
    rf.fit(x_train, y_train)
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), dftrain.columns),reverse=True))
#    importances = rf.feature_importances_
#    indices = np.argsort(importances)[::-1]
#    feat_labels = dftrain.columns
#    for f in range(x_train.shape[1]):
#        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
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
    submission.to_csv("result.csv", index=False)


    window = tk.Tk()
    window.title('违约率预测')

    tk.Label(window, text="测试数据已写入文件，你现在可以输入客户信息预测其违约率", fg='black', font=(12)).grid(columnspan=4,pady=10,sticky=tk.NW)

    # 用户信息
    tk.Label(window, text='性别:', font=(12)).grid(row=2,column=0,pady=10,sticky=tk.NE)
    tk.Label(window, text='年龄:', font=(12)).grid(row=4,column=0,pady=10,sticky=tk.NE)
    tk.Label(window, text='教育程度:', font=(12)).grid(row=6,column=0,pady=10,sticky=tk.NE)
    tk.Label(window, text='收入情况:', font=(12)).grid(row=8,column=0,pady=10,sticky=tk.NE)
    tk.Label(window, text='信用卡数量:', font=(12)).grid(row=10,column=0,pady=10,sticky=tk.NE)
    tk.Label(window, text='成功借款次数:', font=(12)).grid(row=12,column=0,pady=10,sticky=tk.NE)
    tk.Label(window, text='成功还款次数:', font=(12)).grid(row=14,column=0,pady=10,sticky=tk.NE)
    tk.Label(window, text='正常还清次数', font=(12)).grid(row=2,column=2,pady=10,sticky=tk.NE)
    tk.Label(window, text='逾期次数15天内', font=(12)).grid(row=4,column=2,pady=10,sticky=tk.NE)
    tk.Label(window, text='累计借款金额:', font=(12)).grid(row=6,column=2,pady=10,sticky=tk.NE)
    tk.Label(window, text='待还金额:', font=(12)).grid(row=8,column=2,pady=10,sticky=tk.NE)
    tk.Label(window, text='单笔最高借款金额', font=(12)).grid(row=10,column=2,pady=10,sticky=tk.NE)
    tk.Label(window, text='历史最高负债', font=(12)).grid(row=12,column=2,pady=10,sticky=tk.NE)

    a = [[5,2,3,45,6,8,90,4,4,2,5,8,0]]

    var_usr_sex= tk.StringVar()
    entry_usr_sex = ttk.Combobox(window, textvariable=var_usr_sex)
    entry_usr_sex.grid(row=2,column=1)
    entry_usr_sex["values"] = ("男","女")
    entry_usr_sex.current(0)

    var_usr_age= tk.IntVar()
    entry_usr_age = tk.Entry(window, textvariable=var_usr_age).grid(row=4,column=1,pady=10,sticky=tk.NW)

    var_usr_edu= tk.StringVar()
    entry_usr_edu = ttk.Combobox(window, textvariable=var_usr_edu)
    entry_usr_edu.grid(row=6,column=1,pady=10,sticky=tk.NW)
    entry_usr_edu["values"] = ("其他","专科","本科","研究生")
    entry_usr_edu.current(0)
    edu = {"其他":0,"专科":1,"本科":2,"研究生":3}
    eduName = entry_usr_edu.get()

    var_usr_inc= tk.StringVar()
    entry_usr_inc = ttk.Combobox(window, textvariable=var_usr_inc)
    entry_usr_inc.grid(row=8,column=1,pady=10,sticky=tk.NW)
    entry_usr_inc["values"] = ("2000以下","2000-5000","5000-8000","8000-15000","15000以上")
    entry_usr_inc.current(0)
    inc = {"2000以下":0,"2000-5000":1,"5000-8000":2,"8000-15000":3,"15000以上":4}
    incName = entry_usr_inc.get()

    var_usr_credit= tk.IntVar()
    entry_usr_credit = tk.Entry(window, textvariable=var_usr_credit).grid(row=10,column=1,pady=10,sticky=tk.NW)

    var_usr_secceedLendTime= tk.IntVar()
    entry_usr_secceedLendTime= tk.Entry(window, textvariable=var_usr_secceedLendTime).grid(row=12,column=1,pady=10,sticky=tk.NW)

    var_usr_succeedPayTime= tk.IntVar()
    entry_usr_succeedPayTime = tk.Entry(window, textvariable=var_usr_succeedPayTime).grid(row=14,column=1,pady=10,sticky=tk.NW)

    var_usr_payAlready= tk.IntVar()
    entry_usr_payAlready = tk.Entry(window, textvariable=var_usr_payAlready).grid(row=2,column=3,pady=10,sticky=tk.NW)

    var_usr_overDueShort= tk.IntVar()
    entry_usr_overDueShort = tk.Entry(window, textvariable=var_usr_overDueShort).grid(row=4,column=3,pady=10,sticky=tk.NW)

    var_usr_totalPay= tk.DoubleVar()
    entry_usr_totalPay = tk.Entry(window, textvariable=var_usr_totalPay).grid(row=6,column=3,pady=10,sticky=tk.NW)

    var_usr_unpaid= tk.DoubleVar()
    entry_usr_unpaid = tk.Entry(window, textvariable=var_usr_unpaid).grid(row=8,column=3,pady=10,sticky=tk.NW)

    var_usr_onetimeMaxlend= tk.DoubleVar()
    entry_usr_onetimeMaxlend= tk.Entry(window, textvariable=var_usr_onetimeMaxlend).grid(row=10,column=3,pady=10,sticky=tk.NW)

    var_usr_hisMaxLend= tk.DoubleVar()
    entry_usr_hisMaxLend = tk.Entry(window, textvariable=var_usr_hisMaxLend).grid(row=12,column=3,pady=10,sticky=tk.NW)

    var = tk.IntVar()    # 将label标签的内容设置为字符类型，用var来接收hit_me函数的传出内容用以显示在标签上
    tk.Label(window, textvariable=var, fg='black', font=(16), width=30, height=2).grid(row=14,column=3,pady=10,sticky=tk.NW)


    def hit_me():
        if var_usr_sex == "男":
            a[0][0] = 1
        else:a[0][0] = 0
        a[0][1] = var_usr_age.get()
        a[0][2] = edu[eduName]
        a[0][3] = inc[incName]
        a[0][4] = var_usr_credit.get()
        a[0][5] = var_usr_secceedLendTime.get()
        a[0][6] = var_usr_succeedPayTime.get()
        a[0][7] = var_usr_payAlready.get()
        a[0][8] = var_usr_overDueShort.get()
        a[0][9] = var_usr_totalPay.get()
        a[0][10] = var_usr_unpaid.get()
        a[0][11] = var_usr_onetimeMaxlend.get()
        a[0][12] = var_usr_hisMaxLend.get()
        global on_hit
        if on_hit == False:
            on_hit = True
            ans = grid_search.predict_proba(a)
            var.set(ans[0][1])
        else:
            on_hit = False
            var.set('')
 
    # 在窗口界面设置放置Button按键
    b = tk.Button(window, text='点我获取违约可能', font=(12), command=hit_me).grid(row=14,column=2,pady=10,sticky=tk.NW)

    window.mainloop()


if __name__ == "__main__":
    on_hit = False
    main()
