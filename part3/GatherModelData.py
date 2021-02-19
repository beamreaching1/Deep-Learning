# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:59:28 2021

@author: Cayden
"""
lossTestDat = []
accTestDat = []
lossTrainDat = []
accTrainDat = []
paramDat = []
for t in (range(10)):
    t = t+1
    run = 'C:/Users/wagne/Desktop/GitHub/Deep-Learning/part3/m'+str(t)+'.py'
    runfile(run, wdir='C:/Users/wagne/Desktop/GitHub/Deep-Learning/part3')
    lossTestDat.append(lossTest)
    accTestDat.append(accTest)
    paramDat.append(param)
    lossTrainDat.append(lossTrain)
    accTrainDat.append(accTrain)