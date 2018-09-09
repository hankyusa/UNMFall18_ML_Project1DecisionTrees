# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:03:41 2018

@author: Luke Hanks and Trevor La Pay
"""

import argparse
import pandas as pd
import math, operator, functools
import scipy.stats as stats

allFeatIDs = range(60)
featVals = ['A','G','T','C','D','N','S','R']
allClasses = ['N', 'IE', 'EI']

# Flip impurityType to E to run using Entropy Impurity,
# else it will use Gini Index impurity
impurityType = "G"

degreesFreedom = (len(featVals) - 1) * (len(allClasses) - 1)
confidenceLevel = .1
chiSqrThreshold = stats.chi2.ppf(confidenceLevel, degreesFreedom)

totalIncorrect = 0
totalSubtreesStopped = 0

class DecisionNode:
    def __init__(self, inputDF, featIDs, isRoot=False):
        self.isRoot = isRoot
        self.df = inputDF
        self.featIDs = featIDs
        self.children = dict()
        # self.bestClass is the most likely class given classification stops on this node.
        self.bestClass = self.df.groupby('classification').max().iloc[0].name
        self.bestFeat = -1
        if len(list(self.df.classification.value_counts())) > 1 and len(self.featIDs) > 0:
            # The number of unique classes in this set is greater than 1, 
            # therefore impurity is not 0. Also, there are still features to split by.
            # Find the best feature to split by.
            gains = {featID : infoGain(self.df, featID) for featID in self.featIDs}
            self.bestFeat = max(gains, key=gains.get)
            if shouldSplit(self.df, self.bestFeat):
                # Splitting is chi-valuable. Create children.
                childFeatIDs = self.featIDs.copy()
                childFeatIDs.remove(self.bestFeat)
                for featVal in featVals:
                    childDF = getInstances(self.df,self.bestFeat,featVal)
                    if len(childDF) > 0:
                        self.children[featVal] = DecisionNode(childDF, childFeatIDs)
        # self.df = None
    
    def classify(self, dna):
        if self.bestFeat!=-1 and dna[self.bestFeat] in self.children:
            return self.children[dna[self.bestFeat]].classify(dna)
        return self.bestClass
    
    def __str__(self):
        # run print(decTree) to get Mermaid Diagram code.
        s='graph LR\n' if self.isRoot else ''
        s=s+'{}(({}{}))\n'.format(id(self),self.bestClass,' '+str(self.bestFeat) if self.bestFeat!=-1 else '')
        for featVal, child in self.children.items():
            s = s+'{}-- {} -->{}\n'.format(id(self),featVal,id(child))
        for featVal, child in self.children.items():
            s = s + str(child)
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

def impurity(df):
    counts = list(df.classification.value_counts())
    if len(counts) <= 1:
        return 0
    total = sum(counts)
    props = [i/total for i in counts]
    result = 0
    if impurityType == "E":
        for p in props:
            result = result - p * math.log(p, 2)
    else:
        for p in props:
            result += math.pow(p, 2)
        result = 1 - result
    return result

def infoGain(df,featID):
    result = impurity(df)
    for featVal in featVals:
        S_v = getInstances(df,featID,featVal)
        result = result - (len(S_v)/len(df)) * impurity(S_v)
    return result

# Calculate the chi square for a given dataframe.
# Compare the actual number of instances of a class in a
# candidate child node to the expected number given the
# ratio of the classes in the "parent" node.
def getChiSquareForSplit(df, featId):
    chiVal = 0
    # Get count of ALL of parent's instances (regardless of class).
    numParentTotal = len(df)
    for classType in allClasses:
        # Get the set of parent's instances matching classType.
        df_class = getInstances(df, None, None, classType)
        # Get count of parent's instances matching classType.
        numParentObserved = len(df_class)
        if numParentObserved > 0:
            for featureVal in featVals:
                # numClassTotal = total count of number of instances in candidate node
                numClassTotal = len(getInstances(df, featId, featureVal, None))
                # Get the expected value for chi square.
                # This is the number of TOTAL elements in a candidate node * the ratio of
                # the number of observed parent elements for a given class to the total number
                # in the parent.
                expected = numClassTotal * numParentObserved / numParentTotal
                actual = len(getInstances(df, featId, featureVal, classType))
                chiNumerator = math.pow(actual-expected, 2)
                # if expected is zero, this means either the parent has no observed values for that class
                # or the candidate node for that class is empty. In either case, it should not contribute to chi square.
                if (expected > 0):
                    chiVal += chiNumerator/expected
                    # exit early if we've reached the threshold.
                    if (chiVal > chiSqrThreshold):
                        return chiVal
    return chiVal

# Should we split a given node in DecisionTree?
# Use chi square to stop a split if child node distribution is statistically similar
# to parent node. If chi square > critical value, reject null hypothesis that data is
# statistically similar.
def shouldSplit(df, featId):
    if getChiSquareForSplit(df, featId) > chiSqrThreshold:
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

def makeTree(trainingDataFile="training.csv"):
    trainingDF = pd.read_csv(trainingDataFile, header=None, names=['id','features','classification'])
    decTree = DecisionNode(trainingDF, list(allFeatIDs), True)
    return trainingDF, decTree

def testTreeAgainstTrainingData(trainingDF, decTree):
    trainingDF['dtClass'] = trainingDF.apply(lambda i:decTree.classify(i.features), axis=1)
    trainingDF['isCorrect'] = trainingDF.apply(checkCorrectness, axis=1)
    print("Total incorrect: " + str(totalIncorrect))
    print("Percent correct: " + str((len(trainingDF)-totalIncorrect)/len(trainingDF)))
    print("Tree growth stopped via chi-square " + str(totalSubtreesStopped) + " times")
    return functools.reduce(operator.and_,list(trainingDF.isCorrect))

def genterateSubbmissionFile(decTree, testingDataFile="testing.csv", answersDataFile="answers.csv"):
    testingDF = pd.read_csv(testingDataFile, header=None, names=['id','features'])
    testingDF['classification'] = testingDF.features.apply(decTree.classify)
    testingDF.to_csv(answersDataFile,encoding='utf-8',columns=['id','classification'],header=['id','class'],index=False)

def main():
    parser = argparse.ArgumentParser(description='Build a decision tree and test it.')
    parser.add_argument('--training', default="training.csv", type=str,
                        help='The name of the file containing the training data.')
    parser.add_argument('--testing', default="testing.csv", type=str,
                        help='The name of the file containing the testing data.')
    parser.add_argument('--answers', default="answers.csv", type=str,
                        help='The name of the file where the test answers will be put.')
    parser.add_argument('--chiSquarConfidence', default=0.90, type=float,
                        help='Some number between 0 and 1. The level of confidence for the chi squar test.')
    parser.add_argument('--impurity', default="G", type=str,
                        help="How to calculate impurity. 'G' for Gini Index. 'E' for Entropy.")
    args = parser.parse_args()
    trainingDF,decTree = makeTree(args.training)
    testTreeAgainstTrainingData(trainingDF, decTree)
    genterateSubbmissionFile(decTree, args.testing, args.answers)

if __name__ == "__main__": main()
