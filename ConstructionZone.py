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
        self.classification = self.df.groupby('classification').max().iloc[0].name
        self.featIDOfBestGain = -1
        if len(list(self.df.classification.value_counts())) > 1:
            # Entropy is not 0
            gains = {featID : infoGain(self.df, featID) for featID in self.featIDs}
            self.featIDOfBestGain = max(gains, key=gains.get)
            childFeatIDs = self.featIDs.copy()
            childFeatIDs.remove(self.featIDOfBestGain)
            if len(childFeatIDs) != 0:
                # There are more features left to check
                # Create children
                for featVal in featVals:
                    childDF = getInstances(self.df,self.featIDOfBestGain,featVal)
                    if len(childDF) > 0:
                        self.children[featVal] = DecisionNode(childDF, childFeatIDs)
    
    def classify(self, dna):
        if dna[self.featIDOfBestGain] in self.children:
            return self.children[dna[self.featIDOfBestGain]].classify(dna)
        return self.classification
    
    def __str__(self):
        return 'I am a decision tree node.'

def getInstances(df, featID, featVal, classification=None):
    if featID == None or featVal == None:
        return df[df.classification == classification]
    elif classification == None:
        return df[(df.features.str[featID] == featVal)]
    else:
        return df[(df.features.str[featID] == featVal) & (df.classification == classification)]

def entropy(df):
    counts = list(df.classification.value_counts())
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

def checkCorrectness(instance):
    if instance.dtClass == instance.classification:
        return True
    else:
        print('incorect classification:')
        print(instance)
        return False

def makeTree():
    trainingDF = pd.read_csv("training.csv", header=None, names=['id','features','classification'])
    decTree = DecisionNode(trainingDF, list(allFeatIDs))
    return trainingDF, decTree

def testTreeAgainstTrainingData(trainingDF, decTree):
    trainingDF['dtClass'] = trainingDF.apply(lambda i:decTree.classify(i.features), axis=1)
    trainingDF['isCorrect'] = trainingDF.apply(checkCorrectness, axis=1)
    return functools.reduce(operator.and_,list(trainingDF.isCorrect))

def genterateSubbmissionFile(decTree):
    testingDF = pd.read_csv("testing.csv", header=None, names=['id','features'])
    testingDF['classification'] = testingDF.features.apply(decTree.classify)
    testingDF.to_csv('submission.csv', encoding='utf-8',columns=['id','classification'],header=['id','class'],index=False)

def doEverything():
    trainingDF,decTree = makeTree()
    testTreeAgainstTrainingData(trainingDF,decTree)
    genterateSubbmissionFile(decTree)
    return trainingDF,decTree

def doTesting(trainingDF, decTree):
    testTreeAgainstTrainingData(trainingDF, decTree)
    genterateSubbmissionFile(decTree)

# Run the following line if you want to build the tree and test it.
# trainingDF, decTree = doEverything()

# Run the following line if you wnat to test an already built tree.
# doTesting(trainingDF, decTree)
