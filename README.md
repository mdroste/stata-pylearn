
pylearn
=================================

[Overview](#overview)
| [Installation](#installation)
| [Usage](#usage)
| [Benchmarks](#benchmarks)
| [To-Do](#todo)
| [License](#license)

Supervised learning algorithms in Stata with Scikit-Learn

`version 0.3 30mar2020`


Overview
---------------------------------

pylearn is a set of Stata wrapper functions that allows users to estimate many popular supervised learning algorithms (for classification or regression problems) directly from Stata. This program relies on implementations from the Python scikit-learn library, and handles all of the interfacing to and from Python behind the scenes. The result is that you can run random forests, decision trees, boosted decision trees, and neural networks directly from the Stata terminal, as you would with a linear regression model.


Prequisites
---------------------------------

pylearn requires Stata version 16 or higher, since it relies on the [native Python integration](https://www.stata.com/new-in-stata/python-integration/) introduced in Stata 16.0. 

pyforest also requires Python 3+ (though you should really get Python 3+), [scikit-learn](https://scikit-learn.org), and [pandas](https://pandas.pydata.org/). The easiest way to satisfy all of these prerequisites is by installing [Anaconda](https://www.anaconda.com/distribution/#download-section), which contains all of these modules (and many more) out of the box and should be automatically detected by Stata. ALternatively, this repository includes an additional Stata program, pyforest_setup, that will attempt to install these modules automatically for an existing Python installation. 


Installation
---------------------------------

Installing pylearn is very simple.

1. First, install the Stata code and documentation. You can run the following Stata command to install directly from this GitHub repository:

```stata
net install pylearn, from(https://raw.githubusercontent.com/mdroste/stata-learn/master/src) replace
```

2. Install Python if you haven't already, and check to make sure Stata can see it with the following Stata command:
```stata
python query
```

If Stata cannot find your Python installation, refer to the [installation guide](docs/install.md).

3. Make sure that you have the required Python prerequisites installed by running the included Stata program pyforest_setup:

```stata
pylearn_setup
```


Features
---------------------------------

Pyforest currently contains a handful of very popular supervised learning algorithms.


| Function     | Description                               | Related scikit-learn classes                     | 
| ------------ | -----------                               | ------------------------------                    |
| pytree       | Decision trees                           |  [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)<br>[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)    |
| pyforest     | Random forests                            |  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)<br>[randomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)    | 
| pymlp        | Neural networks (multi-layer perceptrons) |  [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)<br>[MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)    |
| pyada        | Boosted trees (AdaBoost.R2)               |  [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)<br>[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)    |
  
Todo
---------------------------------

The following items will be addressed soon:

- [ ] Add support for weights
- [ ] Post-estimation: feature importance (where applicable)
- [ ] Model selection: cross-validation


License
---------------------------------

pylearn is [MIT-licensed](https://github.com/mdroste/stata-pylearn/blob/master/LICENSE).
