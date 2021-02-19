# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 01:38:57 2021

@author: Cayden
!!!!!!!!!!!!!!!!!!!!!!!
Cannot be run without Spyder console
Paste code into spyder console to run
!!!!!!!!!!!!!!!!!!!!!!!
"""

from hessian import hessian
import torch
import scipy.linalg as la

minList = []

for k in range(2):
    runfile('C:/Users/wagne/Desktop/GitHub/Deep-Learning/DeepGradFunction.py', wdir='C:/Users/wagne/Desktop/GitHub/Deep-Learning')
    x = torch.tensor([grad_norm],requires_grad=True)
    h = hessian(x.pow(2).prod(), x, create_graph=True)
    
    h2 = hessian(h.sum(), x)
    
    results = la.eig(h2)
    
    poseig = 0
    for j in results[0]:
        if j > 0:
            poseig += 1
    minratio = poseig / len(results[0])
    minList.append([minratio,lastLoss])