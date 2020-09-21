# -*- coding: utf-8 -*-
import numpy as np
import math
import numpy as np
from collections import Counter
def createDataSet():
    X = []
    Y = []
    filename = 'credit.txt'
    temp_X = []
    data = np.genfromtxt(filename, delimiter=None,dtype=str)
    for n in range(1, len(data)):
        Y.append(data[n][6])
        X_row = []
        for p in range(0, len(data[0])-1):
            X_row.append(data[n][p])
            temp_X.append(data[n][p])
        X.append(X_row)
    X = np.array(X)
    Y = np.array(Y)
    X = X.T
    dataSet = X
    labels = Y
    # change to discrete values
    return dataSet, labels

# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)  # log base 2
    return shannonEnt
# 按特征和特征值划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)  # 这里注意extend和append的区别
    return retDataSet
# ID3决策树分类算法
def chooseBestFeatureToSplitID3(dataSet):
    numFeatures = len(dataSet[0]) - 1  # myDat[0]表示第一行数据, the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # featList是每一列的所有值，是一个列表
        uniqueVals = set(featList)  # 集合中每个值互不相同
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer
# 当所有特征都用完的时候投票决定分类  
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# ID3构建树
def createTreeID3(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 创建类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 如果所有类标签完全相同则停止，直接返回该类标签
    if len(dataSet[0]) == 1:  # 遍历完了所有的标签仍然不能将数据集划分为仅包含唯一类别的分组
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitID3(dataSet)  # 最佳分类特征的下标
    bestFeatLabel = labels[bestFeat]  # 最佳分类特征的名字
    myTree = {bestFeatLabel: {}}  # 创建
    del (labels[bestFeat])  # 从label里删掉这个最佳特征，创建迭代的label列表
    featValues = [example[bestFeat] for example in dataSet]  # 这个最佳特征对应的所有值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTreeID3(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    print(createTreeID3(dataSet,labels))