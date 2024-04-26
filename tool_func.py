import numpy as np
import pandas as pd
from datetime import datetime,timedelta,date
import os

path_trade_date = '/data/disk3/DataBase_stocks/tradeDates/trade_date.csv'
trade_dates = pd.read_csv(path_trade_date)
trade_dates['Date']=pd.to_datetime(trade_dates['Date'])
trade_dates['Date'] = trade_dates['Date'].dt.date      

today=date.today()
format_day='%Y-%m-%d'

def is_trade(day): #输入日期类型
    for i in trade_dates.index:
        if trade_dates.loc[i,'Date']==day:
            return 1
    return 0

def is_first_week(day): #判断是否当周第一个工作日
    if (is_trade(day) and (is_trade(day-timedelta(1))==0)):
        return 1
    else: return 0

def is_last_week(day): #判断是否当周最后一个工作日
    if (is_trade(day) and (is_trade(day+timedelta(1))==0)):
        return 1
    else: return 0

def is_today_first_week(): #判断当天是否当周第一个工作日
    day=date.today()
    if (is_trade(day) and (is_trade(day-timedelta(1))==0)):
        return 1
    else: return 0

def is_today_last_week(): #判断当天是否当周最后一个工作日
    day=date.today()
    if (is_trade(day) and (is_trade(day+timedelta(1))==0)):
        return 1
    else: return 0


#日度因子合成为历史因子
def combine(factors,daily_path=save_daily,factpath=save_factors): 
    files=os.listdir(daily_path)
    files.sort()
    dfs=[]
    for file in files:
        df=pd.read_feather(daily_path+'/'+file)
        df=df.reset_index()
        df.rename(columns={df.columns[0]: 'code'}, inplace=True)
        date=file.split('.')[0]
        df.insert(loc=0, column='date', value=date)
        dfs.append(df)
    df=pd.concat(dfs,axis=0,join='outer')
    for i,factor in enumerate(factors):
        print(factor)
        dfout=df.pivot(index='date', columns='code', values=factor)
        dfout.to_csv(factpath+'/'+factor+'.csv')
        
#entire因子转化为daily（主要用于roll后因子转daily）
def entire_to_daily(trading_day,factors):
    date=int(trading_day.replace('-',''))
    if os.path.exists(save_roll_daily+'/'+str(date)+'.fea'):
        return
    files=os.listdir('/data/disk4/output_stocks/jmchen/factors/minutes/entire_files')
    dfs=[]
    for file in files:
        if any(file.split('.')[0] in factor for factor in factors): #只生成需要roll的daily
            df=pd.read_csv(save_factors+'/'+file).set_index('date')
            dd=df.loc[date,:]
            dd=dd.rename(file.split('.')[0])
            dfs.append(dd)
    df=pd.concat(dfs,axis=1).rename_axis('code')
    df.to_feather(save_roll_daily+'/'+str(date)+'.fea')

#计算2个因子相关系数
def two_factor_IC(path1,path2):
    df1=pd.read_csv(path1).set_index('date')
    df2=pd.read_csv(path2).set_index('date')
    # 计算相关系数并存储
    correlation_values = []
    # 对应列计算相关系数
    for col1, col2 in zip(df1.columns, df2.columns):
        correlation_values.append(df1[col1].corr(df2[col2]))
    correlation_values = [x for x in correlation_values if not math.isnan(x)]
    # 计算相关系数均值
    mean_correlation = sum(correlation_values) / len(correlation_values)
    # mean_correlation=correlation_values.mean()
    # print(sum(correlation_values))
    # 输出相关系数和均值
    print("相关系数均值:", mean_correlation)
    return mean_correlation
