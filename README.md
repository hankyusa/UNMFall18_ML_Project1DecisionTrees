# Machine Learning - Project 1: Decision Trees

## Use

Make sure ConstructionZone.py, training.csv and testing.csv are all in the working directory. Run ConstructionZone.py. If you want to build the tree and test it, run `trainingDF, decTree = doEverything()`. If you wnat to test an already built tree, run `doTesting(trainingDF, decTree)`.

To switch between Entropy and Gini Index impurity, toggle the 
impurityType global (E for entropy, anything else for Gini)

To prevent subtrees from growing in a node if they fail the chi
square threshold test, make sure shouldSplit method call is not
disabled. (since it is not helping accuracy, it needs to be
disabled at the moment)

## Team

Kaggle team name: Hanks, La Pay

Team members:

- Trevor La Pay (graduate)
- Luke Hanks (undergraduate)
