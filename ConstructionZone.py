# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:03:41 2018

@author: Luke Hanks
"""

import pandas as pd
import math

allFeatIDs = range(60)
featVals = ['A','G','T','C','D','N','S','R']

trainingDF = pd.read_csv("smallTraining.csv", header=None, index_col=0, names=['id','features','classification'])

def getInstances(df, featID, featVal, classification=None):
    if featID == None or featVal == None:
        return df[df.classification == classification]
    elif classification == None:
        return df[(df.features.str[featID] == featVal)]
    else:
        return df[(df.features.str[featID] == featVal) & (df.classification == classification)]
    

df1 = getInstances(df=trainingDF, featID=2, featVal='A', classification='N')

def entropy(df):
    counts = list(df['classification'].value_counts())
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

#TODO write the tree class

class DecisionNode:
    def __init__(self, inputDF, featIDs):
        self.df = inputDF
        self.featIDs = featIDs
        self.children = dict()
#        gains = {featID : infoGain(trainingDF, featID) for featID in featIDs}
#        featIDOfBestGain = max(gains, key=gains.get)
        self.featIDOfBestGain = featIDs[0]
        childFeatIDs = featIDs.copy()
        childFeatIDs.remove(self.featIDOfBestGain)
        if len(childFeatIDs) > 0:
            self.classification = None
            # Create children
            for featVal in featVals:
                childDF = getInstances(self.df,self.featIDOfBestGain,featVal)
                if len(childDF) > 0:
                    self.children[featVal] = DecisionNode(childDF, childFeatIDs)
        else:
            self.classification = self.df.groupby('classification').max().iloc[0].name
    
    def classify(self, dna):
        if self.classification == None:
            return self.children[dna[self.featIDOfBestGain]].classify(dna)
        return self.classification

decisionTreeRootNode = DecisionNode(trainingDF, list(allFeatIDs))
print(decisionTreeRootNode.classify('AAAAAAAAAATAGCTGGGCATGGTGGCAGGCGCCTGTAGTTTCAGCTGCTTGGTGTCTGA'))