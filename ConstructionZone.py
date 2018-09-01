# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:03:41 2018

@author: Luke Hanks
"""

import pandas as pd
import math, operator, functools

allFeatIDs = range(60)
featVals = ['A','G','T','C','D','N','S','R']

class DecisionNode:
    def __init__(self, inputDF, featIDs):
        self.df = inputDF
        self.featIDs = featIDs
        self.children = dict()
        gains = {featID : infoGain(self.df, featID) for featID in self.featIDs}
        self.featIDOfBestGain = max(gains, key=gains.get)
#        self.featIDOfBestGain = self.featIDs[0]
        childFeatIDs = self.featIDs.copy()
        childFeatIDs.remove(self.featIDOfBestGain)
        if len(childFeatIDs) != 0:
            self.classification = None
            # Create children
            for featVal in featVals:
                childDF = getInstances(self.df,self.featIDOfBestGain,featVal)
                if len(childDF) > 0:
                    self.children[featVal] = DecisionNode(childDF, childFeatIDs)
        self.classification = self.df.groupby('classification').max().iloc[0].name
    
    def classify(self, dna):
        if dna[self.featIDOfBestGain] in self.children:
            return self.children[dna[self.featIDOfBestGain]].classify(dna)
        return self.classification

def getInstances(df, featID, featVal, classification=None):
    if featID == None or featVal == None:
        return df[df.classification == classification]
    elif classification == None:
        return df[(df.features.str[featID] == featVal)]
    else:
        return df[(df.features.str[featID] == featVal) & (df.classification == classification)]

def entropy(df):
    counts = list(df['classification'].value_counts())
    if len(counts) <= 1:
        return 0
    total = sum(counts)
    props = [i/total for i in counts]
    result = 0
    for p in props:
        result = result - p * math.log(p,2)
    return result

def infoGain(df,featID):
    result = entropy(df)
    for featVal in featVals:
        S_v = getInstances(df,featID,featVal)
        result = result - (len(S_v)/len(df)) * entropy(S_v)
    return result

trainingDF = pd.read_csv("extraSmallTraining.csv", header=None, names=['id','features','classification'])

decisionTreeRootNode = DecisionNode(trainingDF, list(allFeatIDs))

def testDecisionTree(instance):
    return decisionTreeRootNode.classify(instance['features']) == instance['classification']

testResult = functools.reduce(operator.and_,list(trainingDF.apply(testDecisionTree, axis=1)))
print(testResult)

testingDF = pd.read_csv("testing.csv", header=None, names=['id','features'])

def classifyFunction(dna):
    return decisionTreeRootNode.classify(dna)

testingDF['classification'] = testingDF.features.apply(classifyFunction)

testingDF.to_csv('submission.csv', encoding='utf-8',columns=['id','classification'],header=['id','class'],index=False)
