# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:09:05 2021

@author: Cayden Wagner
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import random
from torch.utils.data import Dataset, DataLoader
import pickle

class TrainSet(Dataset):
    def __init__(self):
        self.outCap = json.load(open("MLDS_hw2_1_data/training_label.json"))
        self.dictionary = {}
        self.inFrame = []
        #Default value gets adjusted later
        self.MAXLENGTH = 24

        for a in self.outCap:
            self.inFrame.append(torch.FloatTensor(np.load('MLDS_hw2_1_data/training_data/feat/' + a['id'] + '.npy')))
            for i in a['caption']:   
                temp = i.lower()
                splits = temp.split()
                
                #if len(splits)+2 > self.MAXLENGTH:
                    #self.MAXLENGTH = len(splits)+2
                
                for split in splits:
                    split = split.strip(".!")
                    if split in self.dictionary:
                        self.dictionary[split] += 1
                    else:
                        self.dictionary[split] = 1
                                
        self.dictionary = list(k for k, v in self.dictionary.items() if v > 9)
        
        self.dictionary.insert(0,"<UNK>")
        self.dictionary.insert(0,"<PAD>")
        self.dictionary.insert(0,"<EOS>")
        self.dictionary.insert(0,"<BOS>")
        
        self.indexLookup = {}
        index = 0
        for t in self.dictionary:
            self.indexLookup[t] = index
            index += 1
        
        self.Cap = []
        
        for idx in range(len(self.outCap)):
            
            #Iterates through every sentence in a caption block
            for cap in self.outCap[idx]['caption']:
                tempS = [0]
                temp = cap.lower()
                splits = temp.split()
                splits = splits[0:self.MAXLENGTH-2]
                for split in splits:
                    split = split.strip(".!")
                    
                    if split in self.dictionary:
                        tempS.append(int(self.indexLookup[split]))
                    else:
                        tempS.append(3)
                    
                #insert <EOS> before padding to stop extra processing
                tempS.append(1)
                
                while len(tempS) < self.MAXLENGTH:
                    tempS.append(2)

                self.Cap.append((tempS, idx))
            
    def indexOf(self, word):
        if word in self.dictionary:
            return int(self.indexLookup[word])
        else:
            return 3
        
    def __len__(self):
        return len(self.Cap)

    def __getitem__(self, idx):
        cap, vidIndex = self.Cap[idx]
        cap = torch.LongTensor(cap).view(self.MAXLENGTH, 1)
        oneHot = torch.LongTensor(self.MAXLENGTH, len(self.dictionary))
        oneHot.zero_()
        oneHot.scatter_(1, cap, 1)
        return self.inFrame[vidIndex], oneHot
    

class Att(nn.Module):
    def __init__(self):
        super(Att, self).__init__()

    def forward(self, hiddenState, encoderOut):        
        attOut = torch.bmm(encoderOut.transpose(0,1), hiddenState.transpose(0,1).transpose(1,2))
        attOut = torch.tanh(attOut)
        attW = F.softmax(attOut, dim=1)
        return attW
    
class S2VT(nn.Module):
    def __init__(self, featSize,dictSize,hiddenSize,frameNum,maxCap,batchSize,dropout=0.2):
        super(S2VT, self).__init__()
        self.hiddenSize = hiddenSize
        self.batchSize = batchSize
        self.featSize = featSize
        self.embeddingSize = 512
        self.frameNum = frameNum
        self.maxCap = maxCap
        
        self.att = Att()
        self.lstm1 = nn.LSTM(512, hiddenSize, dropout=dropout)
        self.gru1 = nn.GRU(hiddenSize*2+self.embeddingSize, hiddenSize, dropout=dropout)
        self.embedding = nn.Embedding(dictSize, self.embeddingSize)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(featSize, 512)
        self.out = nn.Linear(hiddenSize, dictSize)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, videoFrames, capFrames):
        lstm1Input = Variable(torch.zeros(self.maxCap, self.batchSize, 512)).cuda();
        gru1Input = Variable(torch.zeros(self.frameNum, self.batchSize, self.hiddenSize+self.embeddingSize)).cuda();
        
        BOS = [0] * self.batchSize
        BOS = Variable(torch.LongTensor([BOS])).resize(batchSize, 1).cuda()
        BOS = self.embedding(BOS)
        
        videoFrames = self.dropout(F.selu(self.fc1(videoFrames)))
        
        lstm1Input = torch.cat((videoFrames, lstm1Input), 0)
        lstm1Output, _ = self.lstm1(lstm1Input)
        
        embedded = self.embedding(capFrames)
        gru1Input = torch.cat((gru1Input, lstm1Output[:self.frameNum,:,:]),2)
        _, hiddenState = self.gru1(gru1Input)

        loss = 0
        for step in range(self.maxCap):
            useTF = True if random.random() <= 0.06 else False
            if step == 0:
                decoderInput = BOS
            elif useTF:
                decoderInput = embedded[:,step-1,:].unsqueeze(1)
            else:
                decoderInput = self.decoderOutput.max(1)[-1].resize(batchSize, 1)
                decoderInput = self.embedding(decoderInput)
            
            attW = self.att(hiddenState, lstm1Output[:self.frameNum])
            tempOut = torch.bmm(attW.transpose(1,2), lstm1Output[:self.frameNum].transpose(0,1))

            gru1Input = torch.cat((decoderInput, lstm1Output[self.frameNum+step].unsqueeze(1), tempOut),2).transpose(0,1)
            
            self.decoderOutput, hiddenState = self.gru1(gru1Input, hiddenState)
            self.decoderOutput = self.softmax(self.out(self.decoderOutput[0]))

            loss += F.nll_loss(self.decoderOutput, capFrames[:,step])

        return loss
    
    def testing(self, videoFrames, dset, pathNum):
        return None
    
    def validing(self, videoFrames, capFrames, vSize):
        
        lstm1Input = Variable(torch.zeros(self.maxCap, vSize, 512)).cuda();
        gru1Input = Variable(torch.zeros(self.frameNum, vSize, self.hiddenSize+self.embeddingSize)).cuda();
        BOS = [0] * vSize
        BOS = Variable(torch.LongTensor([BOS])).resize(vSize, 1).cuda()
        BOS = self.embedding(BOS)
        
        videoFrames = F.selu(self.fc1(videoFrames))
        
        lstm1Input = torch.cat((videoFrames, lstm1Input), 0)
        lstm1Output, _ = self.lstm1(lstm1Input)
        
        gru1Input = torch.cat((gru1Input, lstm1Output[:self.frameNum,:,:]),2)
        _, hiddenState = self.gru1(gru1Input)
        
        loss = 0
        for step in range(self.maxCap):
            if step == 0:
                decoderInput = BOS
            else:
                decoderInput = self.decoderOutput.max(1)[-1].resize(vSize, 1)
                decoderInput = self.embedding(decoderInput)
            attW = self.att(hiddenState, lstm1Output[:self.frameNum])
            tempOut = torch.bmm(attW.transpose(1,2), lstm1Output[:self.frameNum].transpose(0,1))

            gru1Input = torch.cat((decoderInput, lstm1Output[self.frameNum+step].unsqueeze(1), tempOut),2).transpose(0,1)
            
            self.decoderOutput, hiddenState = self.gru1(gru1Input, hiddenState)
            self.decoderOutput = self.softmax(self.out(self.decoderOutput[0]))
            loss += F.nll_loss(self.decoderOutput, capFrames[:,step])
        
        return loss
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

train = None

try:
    datasetFile = open("DatasetPreload", "rb")
    train = pickle.load(datasetFile)
    print("Pickled dataset found, skipping initialization.")
except:
    print("No pickled dataset found, creating from original files.")
    train = TrainSet()
    pickle.dump(train, open("DatasetPreload","wb"))


batchSize = 64

ssm = S2VT(4096,train.__len__(),512,80,train.MAXLENGTH,batchSize)
try:
    print("Model found resuming training.")
    ssm.load_state_dict(torch.load("modelLast"))
except:
    print("No model found training from scratch.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ssm.cuda()

dataloader = DataLoader(train, batch_size=batchSize, shuffle=True, drop_last=True)

ssmOptim = optim.AdamW(ssm.parameters(), lr = 0.001)

vFrames = []
vTarget = []

index = 1400
for row in train.outCap[1400:]:
    for cap in np.unique(np.array(row['caption'])):
        tempCap = [0]
        for word in cap.split():
            word = word.strip(".!").lower()
            tempCap.append(train.indexOf(word))
        if (len(tempCap) + 1) > train.MAXLENGTH:
            continue
        capN = train.MAXLENGTH - (len(tempCap) + 1)
        tempCap += [1]
        tempCap += [2] * capN
        vFrames.append(np.load('MLDS_hw2_1_data/training_data/feat/' + row["id"] + '.npy'))
        vTarget.append(tempCap)
        index += 1
vFrames = Variable(torch.FloatTensor(vFrames).transpose(0,1)).cuda()
vTarget = Variable(torch.LongTensor(vTarget).view(-1, train.MAXLENGTH)).cuda()
vFrames.size(), vTarget.size()


# s2vt
print("S2VT model parameters count: %d" % (ssm.count_parameters()))

epoches = 5
for epoch in range(epoches):
    ssm.train()
    vid = 0
    epoch_losses = 0
    for i, data in enumerate(dataloader):
        ssmOptim.zero_grad()
        frame, oneHotTarget =  Variable(data[0].transpose(0, 1)).cuda(), Variable(data[1]).cuda()
        
        loss = ssm(frame, oneHotTarget.max(2)[-1])
                
        epoch_losses += loss.item() / train.MAXLENGTH
        loss.backward()
        ssmOptim.step()

    ssm.eval()
    betterLoss = 0
    bestLoss = 100
    for i in range(10):
        betterLoss += ssm.validing(vFrames[:,vid:vid+77,:], vTarget[vid:vid+77,:], 77).item() / train.MAXLENGTH
        vid += 77
    if betterLoss < bestLoss:
        bestLoss = betterLoss
        torch.save(ssm.state_dict(), "modelBest")
    print("[Epoch %d] Loss: %f, Better Loss: %f" % (epoch+1, epoch_losses, betterLoss/10/77*64))

torch.save(ssm.state_dict(), "modelLast")