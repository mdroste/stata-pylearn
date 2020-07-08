
pylearn
=================================

[Overview](#overview)
| [Installation](#installation)
| [Usage](#usage)
| [Benchmarks](#benchmarks)
| [To-Do](#todo)
| [License](#license)

Supervised learning in Stata with [scikit-learn](https://scikit-learn.org)

`version 0.6 7jul2020`


Overview
---------------------------------

pylearn is a set of Stata modules that allows Stata users to implement many popular supervised learning algorithms directly from Stata. In particular, pylearn relies on Stata 16.0's [new Python integration](https://www.stata.com/new-in-stata/python-integration/) and the popular Python library [scikit-learn](https://scikit-learn.org) to transfer data and estimates between Stata and a Python interpreter behind the scenes. The result is that users can estimate a broader class of supervised learning algorithms - including decision trees, random forests, boosted trees (gradient boosted trees and AdaBoost), and neural networks (multi-layer perceptrons).

Features
---------------------------------

Pylearn consists of five Stata functions implementing popular supervised ML algorithms:


| Stata Function Name     | Description                               | Related scikit-learn classes                     | 
| ------------ | -----------                               | ------------------------------                    |
| pytree       | Decision trees                          |  [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)<br>[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)    |
| pyforest     | Random forests                            |  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)<br>[randomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)    | 
| pymlp        | Neural networks (multi-layer perceptrons) |  [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)<br>[MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)    |
| pyadaboost   | Adaptive Boosting (AdaBoost)               |  [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)<br>[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)    |
| pygradboost  | Gradient Boosting      |  [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)<br>[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)    |

Each of these programs contains detailed internal documentation. For instance, to view the internal documentation for pyforest, type the following Stata command:
```stata
help pyforest
```


Prequisites
---------------------------------

pylearn requires Stata 16+, since it relies on the [Python integration](https://www.stata.com/new-in-stata/python-integration/) introduced in Stata 16.0. 

pylearn also requires Python 3.6+, [scikit-learn](https://scikit-learn.org), and [pandas](https://pandas.pydata.org/). If you do not have these Python libraries installed, pylearn will try to install them automatically - see the installation section below.



Installation and Upgrading
---------------------------------

Installing pylearn is very simple.

1. First, install the Stata code and documentation. You can run the following Stata command to install everything directly from this GitHub repository:

```stata
net install pylearn, from(https://raw.githubusercontent.com/mdroste/stata-pylearn/master/src) replace
```

2. Install Python if you haven't already, and check to make sure Stata can see it with the following Stata command:
```stata
python query
```

If Stata cannot find your Python installation, refer to the [installation guide](docs/install.md).

3. Make sure that you have the required Python prerequisites installed by running the included Stata program:

```stata
pylearn, setup
```

To check if you have the latest version of pylearn, simply run the following:
```stata
pylearn, upgrade
```


Usage
---------------------------------

Using pylearn is simple, since the syntax for each component looks very similar to other commands for model estimaation in Stata. Notably, calls to pylearn porograms must specify an option called type() that specifies whether the model will be used for classification or regression.

Here is a quick example producing a random forest with the pylearn component pyforest:

```stata
* Load dataset of cars
sysuse auto, clear

* Estimate random forest regression model, training on foreign cars, save predictions as price_predicted
pyforest price mpg trunk weight, type(regress) training(foreign)
predict price_predicted
```

Detailed documentation and usage examples are provided with each Stata file. For instance, to view more examples, type:
```stata
help pyforest
```


Todo
---------------------------------

The following items will be addressed soon:

- [ ] Weights: Add support for weights
- [ ] Post-estimation: return feature importance (where applicable)
- [ ] Model selection: cross-validation
- [ ] Exception handling: more elegant exception handling in Python


License
---------------------------------

pylearn is [MIT-licensed](https://github.com/mdroste/stata-pylearn/blob/master/LICENSE).
