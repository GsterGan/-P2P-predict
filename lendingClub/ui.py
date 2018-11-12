#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
import pickle

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


on_hit = False
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
        
        var.set(a)
    else:
        on_hit = False
        var.set('')
 
# 在窗口界面设置放置Button按键
b = tk.Button(window, text='点我获取违约可能', font=(12), command=hit_me).grid(row=14,column=2,pady=10,sticky=tk.NW)

window.mainloop()


