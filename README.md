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

## Team

Kaggle team name: Hanks, La Pay

Team members:

- Trevor La Pay (graduate)
- Luke Hanks (undergraduate)
