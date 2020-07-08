
pylearn
=================================

[Overview](#overview)
| [Installation](#installation)
| [Usage](#usage)
| [Benchmarks](#benchmarks)
| [To-Do](#todo)
| [License](#license)

Supervised learning algorithms in Stata with Scikit-learn

`version 0.6 7jul2020`


Overview
---------------------------------

pylearn is a set of Stata modules that allows Stata users to implement many popular supervised learning algorithms directly from Stata. In particular, pylearn relies on Stata 16.0's [new Python integration](https://www.stata.com/new-in-stata/python-integration/) and the popular Python library [scikit-learn](https://scikit-learn.org) to transfer data and estimates between Stata and a Python interpreter behind the scenes. The result is that users can estimate a broader class of supervised learning algorithms - including decision trees, random forests, boosted trees (gradient boosted trees and AdaBoost), and neural networks (multi-layer perceptrons).


Prequisites
---------------------------------

pylearn requires Stata version 16 or higher, since it relies on the [native Python integration](https://www.stata.com/new-in-stata/python-integration/) introduced in Stata 16.0. 

pylearn also requires Python 3.5+, [scikit-learn](https://scikit-learn.org), and [pandas](https://pandas.pydata.org/). If you do not have these libraries installed, pylearn will try to install them automatically. 



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

3. Make sure that you have the required Python prerequisites installed by running the included Stata program pylearn_setup:

```stata
pylearn_setup
```


Upgrading
---------------------------------

To check if you have the latest version of pylearn, simply run the following:
```stata
pylearn, upgrade
```


Features
---------------------------------

Pylearn currently contains a handful of very popular supervised learning algorithms.


| Function     | Description                               | Related scikit-learn classes                     | 
| ------------ | -----------                               | ------------------------------                    |
| pytree       | Decision trees                           |  [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)<br>[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)    |
| pyforest     | Random forests                            |  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)<br>[randomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)    | 
| pymlp        | Neural networks (multi-layer perceptrons) |  [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)<br>[MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)    |
| pyadaboost   | Boosted trees               |  [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)<br>[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)    |
| pygradboost  | Gradient boosted trees      |  [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)<br>[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)    |

Each of these programs contains detailed internal documentation. For instance, to view the internal documentation for pyforest, type the following Stata command:
```stata
help pyforest
```
  
Todo
---------------------------------

The following items will be addressed soon:

- [ ] Add support for weights
- [ ] Post-estimation: feature importance (where applicable)
- [ ] Model selection: cross-validation


License
---------------------------------

pylearn is [MIT-licensed](https://github.com/mdroste/stata-pylearn/blob/master/LICENSE).
