#Multi Layer Perceptron

import numpy as np
import math
import random

def activationFunc(v):

    y = 1 / (1 + math.exp(-v))

    return y

def dActivationfuction(y):
    x = y * (1 - y)
    return x

def preprocess(data):
    max = np.max(data, axis=0)
    min = np.min(data, axis=0)
    # print(max)
    # print(min)
    length=max-min

    for i in range(data.__len__()):
        data[i]=(data[i]-min)/length

    return data

def produce(input,w,bias):

    return activationFunc(sum(input*w)+bias)

def feedfoward(input,w,bias,node):
    y = []
    for i in range(input.__len__()):
        tmp=[]
        for j in range(node):
            tmp.append(produce(input[i], w[j], bias[j]))

        if(node==1):
            y.append(tmp[0])
        else:
            y.append(tmp)

    y = np.asarray(y)
    return y

def outputBPG(err,w,bias,deltaW,deltaBias):
    wOutputOld = np.copy(w)
    gradientOutput = (-err * dActivationfuction(y))

    for i in range(x.__len__()):
        for j in range(outputNode):
            w += (gradientOutput[i][j] * x[i] * lr) + (alpha*deltaW)
            deltaW = (gradientOutput[i][j] * x[i] * lr) + (alpha*deltaW)

            bias += (gradientOutput[i][j] * lr) + (alpha*deltaBias)
            deltaBias = (gradientOutput[i][j] * lr) + (alpha*deltaBias)

    return wOutputOld,gradientOutput,deltaW,deltaBias

def hiddenBPG(gradientOutput,w,bias,deltaW,deltaBias):
    wHiddenOld = np.copy(w)
    s=[]
    dAc = dActivationfuction(x)

    for i in range(hiddenNode):
        tmp=[]
        for j in range(outputNode):
            tmp.append(wOutputOld[j][i])
        s.append(tmp)

    sigma=[]

    for k in range(gradientOutput.__len__()):
        tmp=[]
        for i in range(hiddenNode):
            tmp.append(sum(s[i] * gradientOutput[k]))
        sigma.append(tmp)

    gradientHidden=dAc*sigma
    gradientHidden=np.asarray(gradientHidden)


    for i in range(input.__len__()):
        for j in range(hiddenNode):
            w += (gradientHidden[i][j] * input[i] * lr) + (alpha*deltaW)
            deltaW = (gradientHidden[i][j] * input[i] * lr) + (alpha*deltaW)

            bias += (gradientHidden[i][j] * lr) + (alpha*deltaBias)
            deltaBias = (gradientHidden[i][j] * lr) + (alpha*deltaBias)

    return wHiddenOld,gradientHidden,deltaW,deltaBias

def flood_data():
    text_file = open("flooddataset", "r")
    lines = text_file.read().split()
    lines = list(map(float, lines))
    print(lines)

    lines = preprocess(lines)
    print(lines)

    input = []
    dOutput = []
    tmp = []

    for i in range(lines.__len__()):
        if (tmp.__len__() == 8):
            input.append(tmp)
            tmp = []
            dOutput.append(lines[i])
        else:
            tmp.append(lines[i])

    input = np.asarray(input)
    dOutput = np.asarray(dOutput)

    return input,dOutput

def cross_pat():
    text_file = open("cross.pat", "r")
    lines = text_file.read().split()

    for i in lines:
        if("p" in i):
            lines.remove(i)

    lines = list(map(float, lines))

    input = []
    dOutput = []
    tmp = []

    num=0
    for i in range(lines.__len__()):
        tmp.append(lines[i])
        num += 1
        if(num==2):
            input.append(tmp)
            tmp=[]
        if(num==4):
            dOutput.append(tmp)
            tmp=[]
            num=0

    input = np.asarray(input)
    dOutput = np.asarray(dOutput)

    return input , dOutput

alpha=0.1
lr=0.1
layers=2
hiddenNode=3
outputNode=2
wHidden=[]
wOutput=[]
biasHidden=[]
biasOutput=[]

input,dOutput = cross_pat()

dOutput= dOutput.reshape(input.__len__(),outputNode)
print("input :",input.shape)
print("output :",dOutput.shape)

tmp=[]
for i in range(hiddenNode):
    tmp=[]
    for j in range(input[0].__len__()):
        tmp.append(random.uniform(-1, 1))
    wHidden.append(tmp)
    biasHidden.append(random.uniform(-1, 1))

print("HiddenNode :",wHidden.__len__())
print("BiasHiddenNode :",biasHidden.__len__())

for i in range(outputNode):
    tmp=[]
    for j in range(hiddenNode):
        tmp.append(random.uniform(-1, 1))
    wOutput.append(tmp)
    biasOutput.append(random.uniform(-1, 1))
wHidden=np.asarray(wHidden)
wOutput=np.asarray(wOutput)

print("OutputNode :",wOutput.__len__())
print("BiasOutputNode :",biasOutput.__len__())

deltaWOutput=[]
deltaBiasOutput=[]

for i in range(outputNode):
    tmp=[]
    for j in range(hiddenNode):
        tmp.append(0.0)
    deltaWOutput.append(tmp)
    deltaBiasOutput.append(0.0)

deltaWOutput=np.asarray(deltaWOutput)
deltaBiasOutput=np.asarray(deltaBiasOutput)

deltaWHidden = []
deltaBiasHidden = []

for i in range(hiddenNode):
    tmp=[]
    for j in range(input[0].__len__()):
        tmp.append(0.0)
    deltaWHidden.append(tmp)
    deltaBiasHidden.append(0.0)

deltaWHidden=np.asarray(deltaWHidden)
deltaBiasHidden=np.asarray(deltaBiasHidden)

for i in range(100):

    x = feedfoward(input, wHidden, biasHidden, hiddenNode)
    y = feedfoward(x, wOutput, biasOutput, outputNode)
    print(y[0])
    err = dOutput - y

    wOutputOld, gradientOutput, deltaWOutput, deltaBiasOutput = outputBPG(
        err, wOutput, biasOutput, deltaWOutput,deltaBiasOutput)

    wHiddenOld, gradientHidden, deltaWHidden, deltaBiasHidden = hiddenBPG(
        gradientOutput, wHidden, biasHidden,deltaWHidden, deltaBiasHidden)

