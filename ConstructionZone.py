# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:03:41 2018

@author: Luke Hanks
"""

import pandas as pd
import math

allFeatIDs = range(60)
featVals = ['A','G','T','C','D','N','S','R']

trainingDF = pd.read_csv("training.csv", header=None, index_col=0, names=['id','features','label'])

def getInstances(df, featID, featVal, label=None):
    if featID == None or featVal == None:
        return df[df.label == label]
    elif label == None:
        return df[(df.features.str[featID] == featVal)]
    else:
        return df[(df.features.str[featID] == featVal) & (df.label == label)]
    

df1 = getInstances(df=trainingDF, featID=2, featVal='A', label='N')

def entropy(df):
    counts = list(df['label'].value_counts())
    if len(counts) <= 1:
        return 0
    props = [i/sum(counts) for i in counts]
    result = 0
    for p in props:
        result = result - p * math.log(p,2)
    return result

ent = entropy(trainingDF)

def infoGain(df,featID):
    result = entropy(df)
    for featVal in featVals:
        S_v = getInstances(df,featID,featVal)
        result = result - (len(S_v)/len(df)) * entropy(S_v)
    return result

#featIDs = list(allFeatIDs)
#for i in range(60):
#    gains = {featID : infoGain(trainingDF, featID) for featID in featIDs}
#    featIDOfBestGain = max(gains, key=gains.get)
#    featIDs.remove(featIDOfBestGain)
#    print(featIDOfBestGain)

#TODO write the tree class



class DecisionNode:
    def __init__(self, inputDF, featIDs):
        self.df = inputDF
        self.featIDs = featIDs
        self.children = dict()
        gains = {featID : infoGain(trainingDF, featID) for featID in featIDs}
        featIDOfBestGain = max(gains, key=gains.get)
        childFeatIDs = featIDs.copy()
        childFeatIDs.remove(featIDOfBestGain)
        if len(childFeatIDs) > 0:
            for featVal in featVals:
                childDF = getInstances(self.df,featIDOfBestGain,featVal)
                if len(childDF) > 0:
                    self.children[featVal] = DecisionNode(childDF, childFeatIDs)
        print("Node finished!")
        print(self.featIDs)
            
        # Choose a featID
        # Split by feat and make children

decisionTreeRootNode = DecisionNode(trainingDF, list(allFeatIDs))