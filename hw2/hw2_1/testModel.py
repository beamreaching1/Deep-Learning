# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 00:09:58 2021

@author: Cayden Wagner
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
import random
import math

class TrainSet(Dataset):
    def __init__(self):
        self.outCap = json.load(open(sys.argv[1]+"/training_label.json"))
        self.dictionary = set()
        self.inFrame = []
        #Default value gets adjusted later
        self.MAXLENGTH = 22

        for a in self.outCap:
            self.inFrame.append(torch.FloatTensor(np.load(sys.argv[1]+'/training_data/feat/' + a['id'] + '.npy')))
            for i in a['caption']:   
                temp = i.lower()
                splits = temp.split()
                
                if len(splits)+2 > self.MAXLENGTH:
                    self.MAXLENGTH = len(splits)+2
                
                for split in splits:
                    split = split.strip(".!")
                    self.dictionary.add(split)
        
        self.dictionary = list(self.dictionary)
        
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
            for cap in self.outCap[idx]['caption']:
                tempS = [0]
                temp = cap.lower()
                splits = temp.split()
                for split in splits:
                    split = split.strip(".!")
                    
                    if split in self.dictionary:
                        tempS.append(int(self.indexLookup[split]))
                    else:
                        tempS.append(3)

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
        return None
    
    def testing(self, videoFrames, wordDict, pathNum):
        lstm1Input = Variable(torch.zeros(self.maxCap, 1, 512)).cuda();
        gru1Input = Variable(torch.zeros(self.frameNum, 1, self.hiddenSize+self.embeddingSize)).cuda();
        
        BOS = [0]
        BOS = Variable(torch.LongTensor([BOS])).resize(1, 1).cuda()
        BOS = self.embedding(BOS)
        
        videoFrames = F.selu(self.fc1(videoFrames))
        
        lstm1Input = torch.cat((videoFrames, lstm1Input), 0)
        lstm1Output, _ = self.lstm1(lstm1Input)
        
        gru1Input = torch.cat((gru1Input, lstm1Output[:self.frameNum,:,:]),2)
        _, hiddenState = self.gru1(gru1Input)
        
        for step in range(self.maxCap):
            if step == 0:
                attW = self.att(hiddenState, lstm1Output[:self.frameNum])
                tempOut = torch.bmm(attW.transpose(1,2), lstm1Output[:self.frameNum].transpose(0,1))

                gru1Input = torch.cat((BOS, lstm1Output[self.frameNum+step].unsqueeze(1), tempOut),2).transpose(0,1)

                decoderOutput, hiddenState = self.gru1(gru1Input, hiddenState)
                decoderOutput = self.softmax(self.out(decoderOutput[0]))

                softProbs = math.e ** decoderOutput
                bestCanVal, bestCanID = softProbs.topk(pathNum)
                currentScores = bestCanVal.data[0].cpu().numpy().tolist()
                candidates = bestCanID.data[0].cpu().numpy().reshape(pathNum, 1).tolist()
                hiddenStates = [hiddenState] * pathNum
            else:
                newCandidates = []
                for j, candidate in enumerate(candidates):
                    decoderInput = Variable(torch.LongTensor([candidate[-1]])).cuda().resize(1,1)
                    decoderInput = self.embedding(decoderInput)
                    
                    attW = self.att(hiddenState, lstm1Output[:self.frameNum])
                    tempOut = torch.bmm(attW.transpose(1,2), lstm1Output[:self.frameNum].transpose(0,1))

                    gru1Input = torch.cat((decoderInput, lstm1Output[self.frameNum+step].unsqueeze(1), tempOut),2).transpose(0,1)
                    decoderOutput, hiddenStates[j] = self.gru1(gru1Input, hiddenStates[j])
                    decoderOutput = self.softmax(self.out(decoderOutput[0]))
                    
                    softProbs = math.e ** decoderOutput
                    bestCanVal, bestCanID = softProbs.topk(pathNum)

                    for k in range(pathNum):
                        score = currentScores[j] * bestCanVal.data[0, k]
                        new_candidate = candidates[j] + [bestCanID.data[0, k]]
                        newCandidates.append([score, new_candidate, hiddenStates[j]])

                newCandidates = sorted(newCandidates, key=lambda x: x[0], reverse=True)[:pathNum]
                currentScores = [can[0] for can in newCandidates]
                candidates = [can[1] for can in newCandidates]
                hiddenStates = [can[2] for can in newCandidates]
                
        pred = [wordDict.dictionary[wid] if type(wid) == type(1) else wordDict.dictionary[wid.item()] for wid in candidates[0] if wid >= 3]
        
        return pred

train = None

try:
    datasetFile = open("DatasetPreload", "rb")
    train = pickle.load(datasetFile)
    print("Pickled dataset found, skipping initialization.")
except:
    print("No pickled dataset found, creating from original files.")
    train = TrainSet()
    pickle.dump(train, open("DatasetPreload","wb"))


batch_size = 64
ssm = S2VT(4096,train.__len__(),512,80,train.MAXLENGTH,64)
ssm.load_state_dict(torch.load("modelBest-Current"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ssm.cuda()


frames = {}
test_label = json.load(open(sys.argv[1]+ "/testing_label.json"))
for row in test_label:
    frames[row["id"]] = torch.FloatTensor(np.load(sys.argv[1]+'/testing_data/feat/' + row['id'] + '.npy'))

ssm.eval()
predictions = []
indices = []
pathNum = 5
for row in test_label:
    frameIn = Variable(frames[row["id"]].view(-1, 1, 4096)).cuda()
    pred = ssm.testing(frameIn, train, pathNum)
    pred[0] = pred[0].title()
    pred = " ".join(pred)
    predictions.append(pred)
    indices.append(row["id"])

with open(sys.argv[2], 'w') as results:
    for i in range(100):
        results.write(indices[i] + "," + predictions[i] + "\n")