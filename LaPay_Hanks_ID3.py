# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:03:41 2018

@author: Luke Hanks and Trevor La Pay
"""

import argparse
import pandas as pd
import math, operator, functools
import scipy.stats as stats
from collections import Counter
allFeatIDs = range(60)
featVals = ['A','G','T','C','D','N','S','R']
allClasses = ['N', 'IE', 'EI']

totalIncorrect = 0
totalSubtreesStopped = 0
bestFeats = []
class DecisionNode:
    def __init__(self, inputDF, featIDs, impurityFunc, p=0, isRoot=False, fertile=True):
        self.isRoot = isRoot
        self.df = inputDF
        self.featIDs = featIDs
        self.children = dict()
        # self.bestClass is the most likely class given classification stops on this node.
        self.bestClass = self.df.groupby('classification').max().iloc[0].name
        self.bestFeat = -1
        if len(list(self.df.classification.value_counts()))>1 and len(self.featIDs)>0 and len(self.df)>5 and fertile:
            # The number of unique classes in this set is greater than 1, 
            # therefore impurity is not 0. Also, there are still features to split by.
            # Find the best feature to split by.
            gains = {featID : infoGain(self.df, featID, impurityFunc) for featID in self.featIDs}
            self.bestFeat = max(gains, key=gains.get)
            bestFeats.append(self.bestFeat)
            if shouldSplit(self.df, self.bestFeat, p):
                # Splitting is chi-valuable. Create children.
                childFeatIDs = self.featIDs.copy()
                childFeatIDs.remove(self.bestFeat)
                for featVal in featVals:
                    childDF = getInstances(self.df,self.bestFeat,featVal)
                    if len(childDF) > 0:
                        self.children[featVal] = DecisionNode(childDF, childFeatIDs, impurityFunc, p)
        self.df = None
    
    def shouldSpawn(self, childDF, p=0):
        global totalSubtreesStopped
        featValCount = 0
        for featVal in featVals:
            if len(getInstances(self.df, self.bestFeat, featVal)) > 0:
                featValCount += 1
        degreesFreedom = (len(list(self.df.classification.value_counts()))-1)*(featValCount-1)
        chiSqrThreshold = stats.chi2.ppf(p, degreesFreedom)
        chiVal = 0
        # Get count of ALL of parent's instances (regardless of class).
        numParentTotal = len(self.df)
        for classType in allClasses:
            # Get count of parent's instances matching classType.
            numParentObserved = len(getInstances(self.df, classification=classType))
            # If the parent has no observed values for classType, 
            # then classType should not contribute to chiVal.
            if numParentObserved > 0:
                # numChildTotal = total count of number of instances in candidate node
                numChildTotal = len(childDF)
                # If the candidate node has no values,
                # then this candidate should not contribute to chiVal.
                if numChildTotal > 0:
                    # Get the expected value for chi square.
                    # This is the number of TOTAL elements in a candidate node * the ratio of
                    # the number of observed parent elements for a given class to the total number
                    # in the parent.
                    expected = numChildTotal * numParentObserved / numParentTotal
                    actual = len(getInstances(childDF, classification=classType))
                    chiVal += math.pow(actual-expected, 2)/expected
                    # exit early if we've reached the threshold.
                    if (chiVal > chiSqrThreshold):
                        return True
        totalSubtreesStopped += 1
        return False
    
    def classify(self, dna):
        if self.bestFeat != -1 and dna[self.bestFeat] in self.children:
            return self.children[dna[self.bestFeat]].classify(dna)
        return self.bestClass
    
    def __str__(self):
        # run print(decTree) to get Mermaid Diagram code.
        s = 'graph LR\n' if self.isRoot else ''
        s+='{}(({}{}))\n'.format(id(self),self.bestClass,' '+str(self.bestFeat) if self.bestFeat!=-1 else '')
        for featVal, child in self.children.items():
            s += '{}-- {} -->{}\n'.format(id(self),featVal,id(child))
        for featVal, child in self.children.items():
            s += str(child)
        return s

def getInstances(df, featID=None, featVal=None, classification=None):
    if (featID == None or featVal == None) and classification == None:
        return df
    elif featID == None or featVal == None:
        return df[df.classification == classification]
    elif classification == None:
        return df[(df.features.str[featID] == featVal)]
    else:
        return df[(df.features.str[featID] == featVal) & (df.classification == classification)]

def entropy(df):
    classCounts = list(df.classification.value_counts())
    if len(classCounts) <= 1:
        return 0
    total = sum(classCounts)
    proportions = [i/total for i in classCounts]
    result = 0
    for p in proportions:
        result = result - p * math.log(p, 2)
    return result

def giniIndex(df):
    classCounts = list(df.classification.value_counts())
    if len(classCounts) <= 1:
        return 0
    total = sum(classCounts)
    proportions = [i/total for i in classCounts]
    result = 0
    for p in proportions:
        result += math.pow(p, 2)
    result = 1 - result
    return result

def infoGain(df, featID, impurityFunc):
    result = impurityFunc(df)
    for featVal in featVals:
        S_v = getInstances(df,featID,featVal)
        result = result - (len(S_v)/len(df)) * impurityFunc(S_v)
    return result

# Calculate the chi square for a given dataframe.
# Compare the actual number of instances of a class in a
# candidate child node to the expected number given the
# ratio of the classes in the "parent" node.
def getChiSquareForSplit(df, featId, chiSqrThreshold):
    chiVal = 0
    # Get count of ALL of parent's instances (regardless of class).
    numParentTotal = len(df)
    for classType in allClasses:
        # Get count of parent's instances matching classType.
        numParentObserved = len(getInstances(df, classification=classType))
        # If the parent has no observed values for classType, 
        # then classType should not contribute to chiVal.
        if numParentObserved > 0:
            for featureVal in featVals:
                # numChildTotal = total count of number of instances in candidate node
                childDF = getInstances(df, featId, featureVal)
                numChildTotal = len(childDF)
                # If the candidate node has no values,
                # then this candidate should not contribute to chiVal.
                if numChildTotal > 5:
                    # Get the expected value for chi square.
                    # This is the number of TOTAL elements in a candidate node * the ratio of
                    # the number of observed parent elements for a given class to the total number
                    # in the parent.
                    expected = numChildTotal * numParentObserved / numParentTotal
                    #if expected > 5:
                    actual = len(getInstances(childDF, classification=classType))
                    chiNumerator = math.pow(actual - expected, 2)
                    chiVal += chiNumerator/expected
                    # exit early if we've reached the threshold.
                    if (chiVal > chiSqrThreshold):
                        return chiVal
    return chiVal

# Should we split a given node in DecisionTree?
# Use chi square to stop a split if child node distribution is statistically similar
# to parent node. If chi square > critical value, reject null hypothesis that data is
# statistically similar.
def shouldSplit(df, featId, p=0):
    degreesFreedom = (len(allClasses)-1)*(len(featVals)-1)
    chiSqrThreshold = stats.chi2.ppf(p, degreesFreedom)
    if getChiSquareForSplit(df, featId, chiSqrThreshold) > chiSqrThreshold:
        return True
    else:
        global totalSubtreesStopped
        totalSubtreesStopped = totalSubtreesStopped + 1
        return False

def checkCorrectness(instance):
    if instance.dtClass == instance.classification:
        return True
    else:
        # print('incorect classification:')
        # print(instance)
        global totalIncorrect
        totalIncorrect = totalIncorrect + 1
        return False

def makeTree(trainingDataFile="training.csv", impurityFunc=giniIndex, p=0):
    trainingDF = pd.read_csv(trainingDataFile, header=None, names=['id','features','classification'])
    decTree = DecisionNode(trainingDF, list(allFeatIDs), impurityFunc, p, True)
    return trainingDF, decTree

def testTreeAgainstTrainingData(trainingDF, decTree):
    trainingDF['dtClass'] = trainingDF.apply(lambda i:decTree.classify(i.features), axis=1)
    trainingDF['isCorrect'] = trainingDF.apply(checkCorrectness, axis=1)
    print("Total incorrect: " + str(totalIncorrect))
    print("Percent correct: " + str((len(trainingDF)-totalIncorrect)/len(trainingDF)))
    print("Tree growth stopped via chi-square " + str(totalSubtreesStopped) + " times")
    return functools.reduce(operator.and_,list(trainingDF.isCorrect))

def generateSubmissionFile(decTree, testingDataFile="testing.csv", answersDataFile="answers.csv"):
    testingDF = pd.read_csv(testingDataFile, header=None, names=['id','features'])
    testingDF['classification'] = testingDF.features.apply(decTree.classify)
    testingDF.to_csv(answersDataFile,encoding='utf-8',columns=['id','classification'],header=['id','class'],index=False)

def main():
    global impurityType
    parser = argparse.ArgumentParser(description='Build a decision tree and test it.')
    parser.add_argument('--training', default="training.csv", type=str,
                        help='The name of the file containing the training data. Default: training.csv')
    parser.add_argument('--testing', default="testing.csv", type=str,
                        help='The name of the file containing the testing data. Default: testing.csv')
    parser.add_argument('--answers', default="answers.csv", type=str,
                        help='The name of the file where the test answers will be put. Default: answers.csv')
    parser.add_argument('--chiSquareConfidence', default=0.0, type=float,
                        help='Some number between 0 and 1. The level of confidence for the chi-square test. Default: 0.0')
    parser.add_argument('--impurity', default="G", type=str,
                        help="How to calculate impurity. 'G' for Gini Index. 'E' for Entropy. Default: G")
    args = parser.parse_args()
    
    impurityFunc = entropy if args.impurity == 'E' else giniIndex
    
    trainingDF,decTree = makeTree(args.training, impurityFunc, args.chiSquareConfidence)
    testTreeAgainstTrainingData(trainingDF, decTree)
    generateSubmissionFile(decTree, args.testing, args.answers)

if __name__ == "__main__": main()
