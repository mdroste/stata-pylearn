
pylearn
=================================

[Overview](#overview)
| [Features](#features)
| [Prerequisites](#prerequisites)
| [Installation](#installation)
| [Usage](#usage)
| [To-Do](#todo)
| [License](#license)

Supervised learning in Stata with [scikit-learn](https://scikit-learn.org)

`version 0.70 21jul2020`


Overview
---------------------------------

pylearn is a set of Stata modules that allows Stata users to implement many popular supervised learning algorithms - decision trees, random forests, adaptive boosting, gradient boosting, and multi-layer perceptrons (neural networks) - directly from Stata. In particular, pylearn makes use of Stata 16.0's [Python integration](https://www.stata.com/new-in-stata/python-integration/) and the popular Python library [scikit-learn](https://scikit-learn.org) to interface between Stata and Python behind the scenes. 


Features
---------------------------------

Pylearn consists of five Stata functions implementing popular supervised ML algorithms:


| Stata Function Name     | Description                               | Related scikit-learn classes                     | 
| ------------ | -----------                               | ------------------------------                    |
| pytree       | Decision trees                          |  [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)<br>[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)    |
| pyforest     | Random forests                            |  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)<br>[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)    | 
| pymlp        | Neural networks (multi-layer perceptrons) |  [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)<br>[MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)    |
| pyadaboost   | Adaptive Boosting (AdaBoost)               |  [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)<br>[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)    |
| pygradboost  | Gradient Boosting      |  [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)<br>[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)    |

Each of these programs contains detailed internal documentation. For instance, to view the internal documentation for pyforest, type the following Stata command:
```stata
help pyforest
```


Prequisites
---------------------------------

pylearn requires Stata 16+, since it relies on the [Python integration](https://www.stata.com/new-in-stata/python-integration/) introduced in Stata 16.0. 

pylearn also requires Python 3.6+, [scikit-learn](https://scikit-learn.org), and [pandas](https://pandas.pydata.org/). If you do not have these Python libraries installed, pylearn will try to install them automatically - see the installation section below.



Installation
---------------------------------

Installing pylearn is very easy.

1. First, install the Stata code and documentation. You can run the following Stata command to install everything directly from this GitHub repository:
```stata
net install pylearn, from(https://raw.githubusercontent.com/mdroste/stata-pylearn/master/src/) replace
```

2. Install Python, if you haven't already. Make sure that you have the required Python prerequisites installed by running the included Stata program:
```stata
pylearn, setup
```

If Stata cannot find your Python installation, refer to the [installation guide](docs/install.md).

To upgrade to the latest version of pylearn, simply run the following:
```stata
pylearn, upgrade
```


Usage
---------------------------------

Using pylearn is simple, since the syntax for each component looks very similar to other Stata modeling commands.

Here is a quick example of a random forest regression with pylearn's 'pyforest' command:

```stata
* Load auto dataset
sysuse auto, clear

* Estimate random forest regression model, predicting price using mpg, trunk, and weight
* Train only on cars with foreign==1 and test on foreign==0
pyforest price mpg trunk weight, type(regress) training(foreign)
predict price_predicted

* We can also use Stata-like 'if' conditions to specify the training sample, but we won't get out-of-sample RMSE in that case
pyforest price mpg trunk weight if foreign==1, type(regress) 
predict price_predicted_2

* Pyforest chooses defaults automatically, but you can use any hyperparameter scikit-learn can
pyforest price mpg trunk weight, type(regress) training(foreign) max_depth(5) min_samples_split(4)
```

Detailed documentation and usage examples are provided with each Stata file. For instance, see:
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
