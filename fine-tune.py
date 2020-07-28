# -*- coding: utf-8 -*-
"""
Created on 2020/05/13
Updated on 2020/05/13
@author: Li Junyang
"""

#——————————————————————————————————————ANN回归————————————————————————————————————

#———————————————固定随机数———————————————
from numpy.random import seed 
seed(1) 
from tensorflow import set_random_seed 
set_random_seed(1)


#——————————导入所需Python包——————————
import math
import time
import keras
import xlsxwriter
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


#——————————定义常量——————————
model_name = 'ANN_fed'
#STRAT = 3559-2    #对应6月1日20时
#END = 7181-2
#inputsize = 6


#——————————————————————导入数据—————————————————————— 
#df_train = pd.read_csv(open(r'F:\建筑数据集\building data 507\data of office_5.csv'))
#df_train.fillna(method='bfill', inplace=True)    #填充时和后面的保持一致
#
#df_test = pd.read_csv(open(r'F:\建筑数据集\building data 507\data of office_2.csv'))
#df_test.fillna(method='bfill', inplace=True)    #填充时和后面的保持一致

#df = df_train.append(df_test)

df = pd.read_csv(open(r'E:\建筑数据集\building data 507\Office_Cold_6000-9000_6-10_13.csv'))
#Office_NewYork_1000-3000_6-10_7.csv 21936为测试建筑的第一个数据，25586为最后一个数据

#获取时间
Time = np.array(df['hour']).astype(int).reshape(-1)

#获取变量
data = np.array(df.loc[:, ['TemperatureC', 'Humidity', 'hour', 'weekday', 'area', 'energy before', 'energy']])
EC_min = np.min(data[:, -1])
EC_max = np.max(data[:, -1])
normalized_data = (data-np.min(data,0))/(np.max(data,0)-np.min(data,0))


#——————————————————生成训练集、验证集、测试集——————————————————————
#划分训练集、验证集和测试集
x_train =  normalized_data[-3645:-744, :-1]    #6个特征，-3645对应target building 6月1日0时，-744对应10月1日0时
y_train = normalized_data[-3645:-744, -1]    #能耗
time_train = Time[-3645:-744]

##5%
#x_train_5 = x_train[-int(len(x_train)*0.05):]
#y_train_5 = y_train[-int(len(x_train)*0.05):]
#time_train_5 = time_train[-int(len(x_train)*0.05):]
#
##10%
#x_train_10 = x_train[-int(len(x_train)*0.10):]
#y_train_10 = y_train[-int(len(x_train)*0.10):]
#time_train_10 = time_train[-int(len(x_train)*0.10):]
#
##10%
#x_train_15 = x_train[-int(len(x_train)*0.15):]
#y_train_15 = y_train[-int(len(x_train)*0.15):]
#time_train_15 = time_train[-int(len(x_train)*0.15):]
#
##20%
#x_train_20 = x_train[-int(len(x_train)*0.20):]
#y_train_20 = y_train[-int(len(x_train)*0.20):]
#time_train_20 = time_train[-int(len(x_train)*0.20):]
#
##30%
#x_train_30 = x_train[-int(len(x_train)*0.30):]
#y_train_30 = y_train[-int(len(x_train)*0.30):]
#time_train_30 = time_train[-int(len(x_train)*0.30):]
#
##40%
#x_train_40 = x_train[-int(len(x_train)*0.40):]
#y_train_40 = y_train[-int(len(x_train)*0.40):]
#time_train_40 = time_train[-int(len(x_train)*0.40):]
#
##50%
#x_train_50 = x_train[-int(len(x_train)*0.50):]
#y_train_50 = y_train[-int(len(x_train)*0.50):]
#time_train_50 = time_train[-int(len(x_train)*0.50):]
#
##60%
#x_train_60 = x_train[-int(len(x_train)*0.60):]
#y_train_60 = y_train[-int(len(x_train)*0.60):]
#time_train_60 = time_train[-int(len(x_train)*0.60):]
#
##%70
#x_train_70 = x_train[-int(len(x_train)*0.70):]
#y_train_70 = y_train[-int(len(x_train)*0.70):]
#time_train_70 = time_train[-int(len(x_train)*0.70):]
#
##%80
#x_train_80 = x_train[-int(len(x_train)*0.80):]
#y_train_80 = y_train[-int(len(x_train)*0.80):]
#time_train_80 = time_train[-int(len(x_train)*0.80):]
#
##%100
#x_train_100 = x_train[-int(len(x_train)):]
#y_train_100 = y_train[-int(len(x_train)):]
#time_train_100 = time_train[-int(len(x_train)):]
#
x_test =  normalized_data[-744:, :-1]    #6个特征，10月份31天的数据
y_test = normalized_data[-744:, -1]
Y_test = data[-744:, -1]    #冷负荷
time_test = Time[-744:]

#Y_data = data[:, -1]
#time_data = time[:]    #时间
#x_train, x_rest, y_train, y_rest = train_test_split(x_data,y_data,test_size=0.3, random_state=1)
#time_train, time_rest, _, _ = train_test_split(time_data,y_data,test_size=0.3, random_state=1)
#Y_train, Y_rest, _, _ = train_test_split(Y_data,y_data,test_size=0.3, random_state=1)
#
#x_valid, x_test, y_valid, y_test = train_test_split(x_rest,y_rest,test_size=0.5, random_state=1)
#time_valid, time_test, _, _ = train_test_split(time_rest,y_rest,test_size=0.5, random_state=1)
#Y_valid, Y_test, _, _ = train_test_split(Y_rest,y_rest,test_size=0.5, random_state=1)


#——————————————————神经网络回归预测——————————————————
#ann = MLPRegressor(random_state=0) 
#param_grid = {'hidden_layer_sizes':[(math.ceil(x_train.shape[1]*2/3+1)),
#                                    (math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1)),
#                                    (math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1)),
#                                    (math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1)),
#                                    (math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1),math.ceil(x_train.shape[1]*2/3+1))]}
#ANN = GridSearchCV(ann, param_grid, cv=5)    #参数寻优
#ANN = MLPRegressor((5,5), activation='relu', random_state=0, solver='adam')

#ANN = tf.keras.models.load_model('Office_Cold_6000-9000_6-10_13.h5', compile=False)
##ANN = tf.keras.models.load_model('Office_Cold_6000-9000_6-10_13_no_fed.h5', compile=False)
##ANN = tf.keras.models.load_model('fed_model.h5', compile=False)
##y_ANN = ANN.predict(x_test)    #无finetune，直接预测 
#ANN.compile(optimizer='sgd', loss='mse', metrics=['mae'])
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', baseline=None, restore_best_weights=True)
##early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto', baseline=None)
##ANN.fit(x_train_80, y_train_80, batch_size=64, epochs=100, validation_split=0.1, callbacks=[early_stopping])    #模型训练
#y_ANN = ANN.predict(x_test)    #模型预测 
#predictions = y_ANN*(EC_max-EC_min) + EC_min
##predictions = y_ANN
#error = Y_test-predictions
#MAE = np.mean(abs(Y_test-predictions))    #绝对误差
#mae = np.mean(abs(y_test-y_ANN))    #绝对误差
#RMSE = np.sqrt(np.mean(np.square(Y_test-predictions)))    #均方根误差
#rmse = np.sqrt(np.mean(np.square(y_test-y_ANN)))    #均方根误差
#MAPE = np.mean(abs(Y_test-predictions)/Y_test)    #相对百分比误差
#CV_RMSE = RMSE / np.mean(Y_test)
##CPGE = abs(predictions.cumsum()-Y_test.cumsum()) / Y_test.cumsum() * 100    #累积发电量误差
#R2 = 1 - np.sum(np.square(Y_test-predictions)) / np.sum(np.square(Y_test-np.mean(Y_test)))
#print('MAE on test set is %g, RMSE on test set is %g, MAPE on test set is %g, CV_RMSE on test set is %g, R2 on test set is %g.' %(MAE, RMSE, MAPE, CV_RMSE, R2))


# 所有比例数据
MAEs = []
RMSEs = []
MAPEs = []
CV_RMSEs = []
EPOCHs = []
CC = []
P = []

for p in np.linspace(0.01, 1, num=100, endpoint=True):
    print('Current proportion is ' + str(p))
    ANN = tf.keras.models.load_model('Office_Cold_6000-9000_6-10_13.h5', compile=False)
    ANN.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto', baseline=None, restore_best_weights=True)
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto', baseline=None)
    x_train_ = x_train[-int(len(x_train)*p):]
    y_train_ = y_train[-int(len(x_train)*p):]
    time_train_ = time_train[-int(len(x_train)*p):]
    
    start = time.perf_counter()    #计算程序运行时间
    history = ANN.fit(x_train_, y_train_, batch_size=64, epochs=1000, validation_split=0.1, callbacks=[early_stopping])    #模型训练
    end = time.perf_counter()
    t = end - start
    print("Training time is ：", t)
    epoch = len(history.history['loss'])
    y_ANN = ANN.predict(x_test)    #模型预测 
    predictions = y_ANN*(EC_max-EC_min) + EC_min
    #predictions = y_ANN
    #error = Y_test-predictions
    MAE = np.mean(abs(Y_test-predictions))    #绝对误差
    #mae = np.mean(abs(y_test-y_ANN))    #绝对误差
    RMSE = np.sqrt(np.mean(np.square(Y_test-predictions)))    #均方根误差
    #rmse = np.sqrt(np.mean(np.square(y_test-y_ANN)))    #均方根误差
    MAPE = np.mean(abs(Y_test-predictions)/Y_test)    #相对百分比误差
    CV_RMSE = RMSE / np.mean(Y_test)
    #CPGE = abs(predictions.cumsum()-Y_test.cumsum()) / Y_test.cumsum() * 100    #累积发电量误差
    #R2 = 1 - np.sum(np.square(Y_test-predictions)) / np.sum(np.square(Y_test-np.mean(Y_test)))
    MAEs.append(MAE)
    RMSEs.append(RMSE)
    MAPEs.append(MAPE)
    CV_RMSEs.append(CV_RMSE)
    EPOCHs.append(epoch)
    CC.append(t)
    P.append(predictions)
    
PREDICTIONs = np.array(P).T


##——————————————————————保存ANN回归预测结果——————————————————————   
#workbook = xlsxwriter.Workbook('./Prediction of ' + model_name + '.xlsx')
#worksheet = workbook.add_worksheet('Prediction')
#worksheet.write(0, 1, 'Real data')
#worksheet.write(0, 2, 'Prediction data')
#worksheet.write(0, 3, 'distacne')
#worksheet.write(0, 4, 'error')
#worksheet.write(0, 5, 'MAE of prediction')
#worksheet.write(0, 6, 'RMSE of prediction')
#worksheet.write(0, 7, 'MAPE of prediction')
#worksheet.write(0, 8, 'CV_RMSE of prediction')
#worksheet.write(0, 9, 'R2 of prediction')
#worksheet.write(1, 5, MAE)
#worksheet.write(1, 6, RMSE)
#worksheet.write(1, 7, MAPE)
#worksheet.write(1, 8, CV_RMSE)
#worksheet.write(1, 9, R2)
#for i in range(predictions.shape[0]):
#    worksheet.write(i+1, 1, Y_test[i])
#    worksheet.write(i+1, 2, predictions[i])
#    worksheet.write(i+1, 3, dis[i])
#    worksheet.write(i+1, 4, error[i])
#    #worksheet.write(i+1, 6, CPGE[i])
#workbook.close()


##——————————————————————可视化——————————————————————   
#plt.close('all')
#x = range(len(predictions[:240]))
#names = time_test[:240].tolist()[::4]
#pos = x[::4]
#plt.figure(figsize=(6.4*3.2,4.8*3.2), dpi=100)
#plt.plot(x, Y_test[:240], 'b-', label='Actual value')
#plt.plot(x, predictions[:240], 'r--', label='Forecasting value')
#plt.xticks(pos, names, rotation=80)
#plt.tick_params(labelsize=15)
#plt.xlabel('Time', fontsize=20)
#plt.ylabel('Cooling load(kW)', fontsize=20)
#plt.legend(fontsize=20)
#plt.savefig('Prediction of ' + model_name + '.png')
#plt.show()

#
#y_ANN = ANN.predict(x_train)    #模型预测
#x = range(len(y_ANN[:100]))
#names = time_train[:100].tolist()[::2]
#pos = x[::2]
#plt.figure(figsize=(6.4*3.2,4.8*3.2), dpi=100)
#plt.plot(x, y_train[:100], 'b-', label='Actual value')
#plt.plot(x, y_ANN[:100], 'r--', label='Forecasting value')
#plt.xticks(pos, names, rotation=80)
#plt.tick_params(labelsize=15)
#plt.xlabel('Time', fontsize=20)
#plt.ylabel('Cooling load(kW)', fontsize=20)
#plt.legend(fontsize=20)
##plt.savefig('Prediction of ' + model_name + '.png')
#plt.show()