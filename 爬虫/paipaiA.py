from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import re
import csv
import numpy as np
from bs4 import BeautifulSoup
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
driver = webdriver.Chrome()
ii=127959589
def LoginRRD(username, password):
    try:
        print(u'准备登陆人人贷网站...')
        driver.get("https://ac.ppdai.com/User/Login?message=&redirect=")
        elem_user = driver.find_element_by_name("UserName")
        elem_user.send_keys(username)
        elem_pwd = driver.find_element_by_name("Password")
        elem_pwd.send_keys(password)
        elem_pwd.send_keys(Keys.RETURN)
        time.sleep(15) #这里设置睡眠时间，是为了使浏览器有时间保存cookies
        print(u'登录 successful!')

    except Exception as e:
        print("Error:", e)
    finally:
        print(u'End Login!\n')


def parse_userinfo(loanid):#自定义解析借贷人信息的函数
    urll="https://invest.ppdai.com/loan/info/%s"  % loanid
    #这个urll我也不知道怎么来的，貌似可以用urll="http://www.we.com/loan/%f" % loanid+timestamp (就是页面本身
    try:
        s=driver.get(urll)
        #print(driver.page_source)
        time.sleep(2) 
        html = BeautifulSoup(driver.page_source.encode('utf-8'),'lxml')
        #print(html)
        #s= html.find('section',class_='content')
        s= html.find('section',id='borrowerInfo')
        #print(s)
        li=[]
        li.append(ii)
        ss=s.find_all('span')
        print(len(ss))
        f=1
        for i in range(1,13):
            l=str(ss[f].contents[0]).strip()
            li.append(l)
            f=f+2
        print(li)
        
    except Exception as e:
         print("Error:", e)
         #driver.quit()
    

    return 0
def parse_userinfo2(loanid):#自定义解析借贷人信息的函数
    urll="https://invest.ppdai.com/loan/info/%s"  % loanid
    #这个urll我也不知道怎么来的，貌似可以用urll="http://www.we.com/loan/%f" % loanid+timestamp (就是页面本身
    try:
        s=driver.get(urll)
        time.sleep(1) 
        #print(driver.page_source)
        html = BeautifulSoup(driver.page_source.encode('utf-8'),'lxml')
        #print(html)
        #s= html.find('section',class_='content')
        s= html.find('section',id='borrowerInfo')
        #print(s)
        li=[]
        li.append(ii)
        ss=s.find_all('span')
        #print(len(ss))
        f=1
        for i in range(1,13):
            l=str(ss[f].contents[0]).strip()
            li.append(l)
            f=f+2
        s2= html.find('section',id='loanRecord')
        #print(s)
        
        ss=s2.find_all('span')
        #print(len(ss))
        f=1
        for i in range(1,4):
            l=str(ss[f].contents[0]).strip()
            li.append(l)
            f=f+2
        s3= html.find('section',id='repayRecord')
        #print(s)
    
       
        ss=s3.find_all('span')
       # print(len(ss))
        f=1
        for i in range(1,7):
            l=str(ss[f].contents[0]).strip()
            li.append(l)
            f=f+2
        s4= html.find('section',id='debtRecord')
        #print(s)
       
        ss=s4.find_all('span')
        #print(len(ss))
        f=1
        for i in range(1,5):
            l=str(ss[f].contents[0]).strip()
            li.append(l)
            f=f+2
        print(li)
        
        
    except Exception as e:
         print("Error:", e)
         #driver.quit()
    

    return 0
username = "18889347177"
password = u"a12345678"

LoginRRD(username, password)
for i in range(0,1):
    try:
        r=0
        parse_userinfo(ii)
    except Exception as e:
        print("Error:", e)
    ii=ii+1
    time.sleep(10)
for i in range(0,100):
    try:
        r=0
        parse_userinfo2(ii)
    except Exception as e:
        print("Error:", e)
    ii=ii+1
    

