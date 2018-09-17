# Machine Learning - Project 1: Decision Trees

## Usage

Programmed for Python 3.6.6, but may work with other versions of Python 3. 

Requires the following libraries.

* argparse
* pandas
* math
* operator
* functools
* scipy
* collections

Make sure LaPay_Hanks_ID3.py, training.csv and testing.csv are all in the working directory.

To run with defaults simply navigate to the working directory in your terminal and execute:

```
$ python LaPay_Hanks_ID3.py
```

The following opptions are available.

```
usage: LaPay_Hanks_ID3.py [-h] [--training TRAINING] [--testing TESTING]
                          [--answers ANSWERS]
                          [--chiSquareConfidence CHISQUARECONFIDENCE]
                          [--impurity IMPURITY]

Build a decision tree and test it.

optional arguments:
  -h, --help            show this help message and exit
  --training TRAINING   The name of the file containing the training data.
                        Default: training.csv
  --testing TESTING     The name of the file containing the testing data.
                        Default: testing.csv
  --answers ANSWERS     The name of the file where the test answers will be
                        put. Default: answers.csv
  --chiSquareConfidence CHISQUARECONFIDENCE
                        Some number between 0 and 1. The level of confidence
                        for the chi-square test. Default: 0.0
  --impurity IMPURITY   How to calculate impurity. 'G' for Gini Index. 'E' for
                        Entropy. Default: G
```

Note: Unambiguous abreviations work.

### Mermaid Diagrams

The `__str__` method in the DecisionNode class generates [Mermaid diagram](https://mermaidjs.github.io/) code. In the diagrams each node has a classification it would give if the search stopped there. The class is followed by the feature that the node decided to split by. The edges are marked by the feature value of the parent node's splitting feature that produced the child node. If a leaf node has a feature, then that node was prevented from splitting based on the chi squared method. If a leaf node has no feature, then it is a true dead end where impurity reached 0.

## Team

Kaggle team name: Hanks, La Pay

Team members:

- Trevor La Pay (graduate)
- Luke Hanks (undergraduate)
