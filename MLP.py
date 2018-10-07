#Multi Layer Perceptron
from numpy import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt

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

    # for i in range(input.__len__()):
    #     tmp=[]
    #     for j in range(node):
    #         tmp.append(produce(input[i], w[j], bias[j]))
    #
    #     if(node==1):
    #         y.append(tmp[0])
    #     else:
    #         y.append(tmp)
    #
    # y = np.asarray(y)

    for i in range(node):
        y.append(produce(input, w[i], bias[i]))

    y = np.asarray(y)

    return y

def outputBPG(err,w,bias,deltaW,deltaBias):

    gradientOutput = (-err * dActivationfuction(y))
    wOutputOld = []

    for i in range(x.__len__()):
        wOutputOld.append(w.copy())

        for j in range(outputNode):
            w[j] += (gradientOutput[i][j] * x[i] * lr) + (alpha*deltaW[j])
            deltaW[j] = (gradientOutput[i][j] * x[i] * lr) + (alpha*deltaW[j])

            bias[j] += (gradientOutput[i][j] * lr) + (alpha*deltaBias[j])
            deltaBias[j] = (gradientOutput[i][j] * lr) + (alpha*deltaBias[j])

    wOutputOld=np.asarray(wOutputOld)

    return wOutputOld,gradientOutput,deltaW,deltaBias

def hiddenBPG(gradientOutput,w,bias,deltaW,deltaBias):
    dAc = dActivationfuction(x)
    s = []

    for k in range(wOutputOld.__len__()):
        s.append([])
        for i in range(hiddenNode):
            tmp=[]
            for j in range(outputNode):
                tmp.append(wOutputOld[k][j][i])
            s[-1].append(tmp)

    sigma=[]

    for k in range(gradientOutput.__len__()):
        tmp=[]
        for i in range(hiddenNode):
            tmp.append(sum(s[k][i] * gradientOutput[k]))

        sigma.append(tmp)

    gradientHidden=dAc*sigma
    gradientHidden=np.asarray(gradientHidden)

    for i in range(input.__len__()):
        for j in range(hiddenNode):
            w[j] += (gradientHidden[i][j] * input[i] * lr) + (alpha*deltaW[j])
            deltaW[j] = (gradientHidden[i][j] * input[i] * lr) + (alpha*deltaW[j])

            bias[j] += (gradientHidden[i][j] * lr) + (alpha*deltaBias[j])
            deltaBias[j] = (gradientHidden[i][j] * lr) + (alpha*deltaBias[j])

    return gradientHidden,deltaW,deltaBias

def bpg(err,wOutput,deltaWOutput,deltaBiasOutput,wHidden,deltaWHidden,deltaBiasHidden,input):

    gradientOutput = (err * dActivationfuction(y))
    wOutputOld = wOutput.copy()

    # print("1",wOutput)
    # print("2",gradientOutput)
    # print("3",x)
    # print("4",deltaWOutput)
    # print("5",biasOutput)

    for i in range(gradientOutput.__len__()):
        wOutput[i] += (gradientOutput[i] * x * lr) + (alpha * deltaWOutput[i])
        deltaWOutput[i] = (gradientOutput[i] * x * lr) + (alpha * deltaWOutput[i])

        biasOutput[i] += (gradientOutput[i] * lr) + (alpha * deltaBiasOutput[i])
        deltaBiasOutput[i] = (gradientOutput[i] * lr) + (alpha * deltaBiasOutput[i])

    # print(wOutput)

    dAc = dActivationfuction(x)
    sigma=[]

    for i in range(hiddenNode):
        tmp=[]
        for j in range(outputNode):
            tmp.append(wOutputOld[j][i]*gradientOutput[j])
        sigma.append(sum(tmp))

    gradientHidden = dAc*sigma

    for i in range(gradientHidden.__len__()):
        wHidden[i] += (gradientHidden[i] * input * lr) + (alpha * deltaWHidden[i])
        deltaWHidden[i] = (gradientHidden[i] * input * lr) + (alpha * deltaWHidden[i])

        biasHidden[i] += (gradientHidden[i] * lr) + (alpha * deltaBiasHidden[i])
        deltaBiasHidden[i] = (gradientHidden[i] * lr) + (alpha * deltaBiasHidden[i])


    return wOutput,wHidden,deltaWOutput,deltaWHidden

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
        tmp.append(lines[i])

        if (tmp.__len__() == 9):
            input.append(tmp)
            tmp = []

    random.shuffle(input)

    for i in range(input.__len__()):
        dOutput.append(input[i].pop())

    input = np.asarray(input)
    dOutput = np.asarray(dOutput)

    return input,dOutput,1

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

    return input , dOutput , 2

def flood_acc(y,dOutput):
    mse = ((dOutput - y) ** 2).mean(axis=None)

    return mse

def cross_acc(y,dOutput):

    tmp=[]
    if(y[0]>y[1]):
        tmp=[1.0,0.0]
    elif(y[0]<y[1]):
        tmp=[0.0,1.0]

    if(np.array_equal(dOutput,tmp)):
        return 1
    else:
        return 0

alpha=0.5
lr=0.1
layers=2
hiddenNode=3
# input,dOutput , outputNode = cross_pat()
inputRaw,dOutputRaw , outputNode = cross_pat()

dOutputRaw= dOutputRaw.reshape(inputRaw.__len__(),outputNode)
print("input :",inputRaw.shape)
print("output :",dOutputRaw.shape)

accall=[]
for n in range(10):
    wHidden = []
    wOutput = []
    biasHidden = []
    biasOutput = []

    tmp=[]
    for i in range(hiddenNode):
        tmp=[]
        for j in range(inputRaw[0].__len__()):
            tmp.append(random.uniform(-1,1))
        wHidden.append(tmp)
        biasHidden.append(random.uniform(-1,1))

    # print("HiddenNode :",wHidden.__len__())
    # print("BiasHiddenNode :",biasHidden.__len__())

    for i in range(outputNode):
        tmp=[]
        for j in range(hiddenNode):
            tmp.append(random.uniform(-1,1))
        wOutput.append(tmp)
        biasOutput.append(random.uniform(-1,1))
    wHidden=np.asarray(wHidden)
    wOutput=np.asarray(wOutput)

    # print("OutputNode :",wOutput.__len__())
    # print("BiasOutputNode :",biasOutput.__len__())

    # train set validation set test set
    percent = int((inputRaw.__len__())*(10/100))

    inputTest = inputRaw[n*percent:percent*(n+1)]
    dOutputTest = dOutputRaw[n*percent:percent*(n+1)]

    input = concatenate((inputRaw[0:n * percent],inputRaw[percent * (n + 1):-1]),axis=0)
    dOutput = concatenate((dOutputRaw[0:n * percent],dOutputRaw[percent * (n + 1):-1]),axis=0)

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

    epoch=[100]
    acc=[]
    for k in range(epoch.__len__()):

        for j in range(epoch[k]):
            tmp=0
            for i in range(input.__len__()):

                x = feedfoward(input[i], wHidden, biasHidden, hiddenNode)
                y = feedfoward(x, wOutput, biasOutput, outputNode)

                tmp+=cross_acc(y,dOutput[i]) # uncomment if use cross_pat

                # tmp+=flood_acc(y,dOutput[i]) # uncomment if use flood data set

                err = dOutput[i] - y

                wOutput, wHidden, deltaWOutput, deltaWHidden = bpg(
                    err,wOutput,deltaWOutput,deltaBiasOutput,wHidden,deltaWHidden,deltaBiasHidden,input[i])

            # show accuracy
            # print(j,tmp,"/",input.__len__(),end=" = ")
            acc.append(tmp/input.__len__())
            # print(acc[-1])
    acc=asarray(acc)
    tmp=acc.argmin(axis=0)
    print(n,"ACC = ",acc[0],acc[-1],acc[tmp],tmp) # show accuracy between first and last epoch

    # t = np.arange(0,epoch[0])
    # plt.plot(t, acc, 'r') # plotting t, a separately
    # plt.show()

    # Test The Model
    err=0

    for i in range(inputTest.__len__()):
        x = feedfoward(inputTest[i], wHidden, biasHidden, hiddenNode)
        y = feedfoward(x, wOutput, biasOutput, outputNode)

        # err += flood_acc(y, dOutputTest[i])
        err += cross_acc(y, dOutputTest[i])

    # print("ERR = ",err) # for flood

    err=err/inputTest.__len__() # for cross
    print("ACC = ",err) # for cross
    accall.append(err)
# print("Over All Error = ",sum(accall)/accall.__len__()) # for flood
print("Over All Accuracy = ",sum(accall)/accall.__len__()) # for cross