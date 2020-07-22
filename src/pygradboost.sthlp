{smcl}
{* *! version 0.70 21jul2020}{...}
{viewerjumpto "Syntax" "pygradboost##syntax"}{...}
{viewerjumpto "Description" "pygradboost##description"}{...}
{viewerjumpto "Options" "pygradboost##options"}{...}
{viewerjumpto "Stored results" "pygradboost##results"}{...}
{viewerjumpto "Examples" "pygradboost##examples"}{...}
{viewerjumpto "Author" "pygradboost##author"}{...}
{viewerjumpto "Acknowledgements" "pygradboost##acknowledgements"}{...}
{title:Title}
 
{p2colset 5 20 26 2}{...}
{p2col :{hi:pygradboost} {hline 2}}Gradient boosted trees with Python and scikit-learn{p_end}
{p2colreset}{...}
 

{marker syntax}{title:Syntax}
 
{p 8 15 2}
{cmd:pygradboost} {depvar} {indepvars} {ifin}, type(string) [{cmd:}{it:options}]
                               
{synoptset 32 tabbed}{...}
{synopthdr :options}
{synoptline}
{syntab :Main}
{synopt :{opt type(string)}}{it:string} may be {bf:regress} or {bf:classify}.{p_end}

{syntab :Pre-processing}
{synopt :{opt training(varname)}}varname is an indicator for the training sample{p_end}

{syntab :Gradient Boosting options}
{synopt :{opt n_estimators(#)}}Number of boosting stages{p_end}
{synopt :{opt loss(string)}}Loss function when updating weights{p_end}
{synopt :{opt learning_rate(#)}}Shrinks the contribution of each predictor{p_end}
{synopt :{opt criterion(string)}}Criterion for splitting nodes (see details below){p_end}
{synopt :{opt max_depth(#)}}Maximum tree depth{p_end}
{synopt :{opt min_samples_split(#)}}Minimum observations per node{p_end}
{synopt :{opt min_weight_fraction_leaf(#)}}Minimum fraction of obs at each leaf{p_end}
{synopt :{opt max_features(numeric)}}Maximum number of features to consider per tree{p_end}
{synopt :{opt max_leaf_nodes(#)}}Maximum leaf nodes{p_end}
{synopt :{opt min_impurity_decrease(#)}}Propensity to split{p_end}
{synopt :{opt subsample(#)}}Fraction of obs to use for fit trees; stochastic gradient boosting if <1{p_end}

{syntab :Early Stopping Options}
{synopt :{opt n_iter_no_change(#)}}Stops early if # iterations without change in fit on a validation subsample{p_end}
{synopt :{opt validation_fraction(#)}}Fraction of training data to set aside for validation when n_iter_no_change specified{p_end}
{synopt :{opt tol(#)}}Tolerance threshold for early stopping when n_iter_no_change specified{p_end}

{synoptline}
{p 4 6 2}
{p_end}
    For more information on syntax and options, see the {browse "https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting":scikit-learn documentation for Gradient Boosting}.
    {browse "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html":Gradient boosting regression syntax}
    {browse "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html":Gradient boosting classification syntax}
 

{marker description}{...}
{title:Description}
 
{pstd}
{opt pygradboost} performs regression or classification with gradient boosted decision trees; a generalization of boosted trees to arbitrary differentiable loss functions. 

{pstd} {opt pygradboost} relies critically on the Python integration functionality introduced with Stata 16. Therefore, users will need Stata 16, Python 3.6+, and the scikit-learn Python library installed in order to run. 

{pstd} For more information on pytree, refer to the pylearn GitHub page: {browse "https:/www.github.com/mdroste/stata-pylearn":https:/www.github.com/mdroste/stata-pylearn}.

 
{marker options}{...}
{title:Options}
 
{dlgtab:Main}
 
{phang}
{opth type(string)} declares whether this is a regression or classification problem. In general, type(classify) is more appropriate when the dependent variable is categorical, and type(regression) is more appropriate when the dependent variable is continuous.


{dlgtab:Pre-processing}

{phang}
{opt training(varname)} identifies an indicator variable in the current dataset that is equal to 1 when an observation should be used for training and 0 otherwise. If this option is specified, frac_training() is ignored.


{dlgtab:Adaptive boosting options}
 
{phang}
{opt n_estimators(#)} determines the number of boosting stages to perform. 

{phang}
{opt loss(string)} specifies the loss function. When type(classify) is specified, valid options are loss(exponential) and loss(deviance). When type(regress) is specified, valid options are loss(ls), loss(lad), loss(huber), or loss(quantile). The default is loss(deviance) for classification and loss(ls) for regression.

{phang}
{opt learning_rate(#)} learning rate shrinks the contribution of each tree by learning_rate.

{phang}
{opt criterion} specifies the criterion for determining the quality of a split. Valid options are criterion(friedman_mse) (mean squared error with improvement score by Friedman), criterion(mse), and criterion(mae). Default is criterion(friedman_mse).

{phang}
{opt max_depth(#)} specifies the maximum tree depth. Default: max_depth(3).

{phang}
{opt min_samples_split(#)} specifies the minimum number of observations required to consider splitting an internal node of a tree. Default: min_samples_split(2).

{phang}
{opt min_samples_leaf(#)} specifies the minimum number of observations required at each 'leaf' node of a tree. Default: min_samples_leaf(1).

{phang}
{opt min_weight_fraction_leaf(#)} specifies the minimum weighted fraction of the sum of weights required at each leaf node. When weights are not specified, this is simply the minimum fraction of observations required at each leaf node. Default: min_weight_fraction_leaf(0).

{phang}
{opt max_features(string)} specifies the number of features to consider when looking for the best split. Default: max_features(# of features/independent variables). Other options are max_features(sqrt) (the square root of the number of features), max_features(log2) (the base-2 logarithm of the number of features), an integer, or a float. If a non-integer is specified, then int(max_features,number of features) are considered at each split. 

{phang}
{opt max_leaf_nodes(#)} Grow trees with max_leaf_nodes in best-first fashion, where best is defined in terms of relative reduction in impurity. Default: max_leaf_nodes is unconstrained (infinite).

{phang}
{opt min_impurity_decrease(#)} determines the threshold such tha a node is split if it induces a decrease in impurity criterion greater than or equal to this value. Default: mi_impurity_decrease(0).

{phang}
{opt subsample(#)} fraction of observations to use for fitting base learners. If smaller than 1, we are doing stochastic gradient boosting. Must be between 0 and 1. Default: subsample(1).


{dlgtab:Early stopping options}

{phang}
{opt n_iter_no_change(#)} is used to decide if early stopping terminates training if the model fit on a validation set is not improving with successive iterations. By default, n_iter_no_change is not specified, and no early stopping is implemented. If specified, it will set aside a fraction of the training data specified by validation_fraction(#), default 0.1, and terminate training if the validation score does not improve.

{phang}
{opt validation_fraction(#)} is a proportion of the training data to set aside for early stopping. Must be between 0 and 1, and only used if n_iter_no_change() (see above) is specified. By default, validation_fraction(0.1) is used.

{phang}
{opt tol(#)} is a tolerance threshold for early stopping. When the loss is not improving by at least tol(#) for n_iter_no_change iterations (see above), training stops. Default is tol(0.0001).


{marker results}{...}
{title:Stored results}

{synoptset 24 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:e(N)}}number of observations in training sample{p_end}
{synopt:{cmd:e(N_test)}}number of observations in test sample{p_end}
{synopt:{cmd:e(K)}}number of features{p_end}
{synopt:{cmd:e(training_rmse)}}root mean squared error on training data, if type(regress){p_end}
{synopt:{cmd:e(test_rmse)}}root mean squared error on test data, if type(regress) and training() specified{p_end}
{synopt:{cmd:e(training_mae)}}mean absolute error on training data, if type(regress){p_end}
{synopt:{cmd:e(test_mae)}}mean absolute error on test data, if type(regress) and training() specified{p_end}
{synopt:{cmd:e(training_accuracy)}}accuracy on training data, if type(classify){p_end}
{synopt:{cmd:e(test_accuracy)}}accuracy on test data, if type(classify) and training() specified{p_end}

{synoptset 24 tabbed}{...}
{syntab:Macros}
{synopt:{cmd:e(features)}}List of feature names{p_end}


{marker examples}{...}
{title:Examples}

{pstd}{bf:Example 1}: Classification with gradient boosting, saving predictions as a new variable called iris_prediction{p_end}
{phang2} Load data{p_end}
{phang3} {stata webuse iris, clear}{p_end}
{phang2} Run adaptive boosting classifier{p_end}
{phang3} {stata pygradboost iris seplen sepwid petlen petwid, type(classify)}{p_end}
{phang2} Save predictions in a variable called iris_hat{p_end}
{phang3} {stata predict iris_hat}{p_end}

{pstd}{bf:Example 2}: Classification with gradient boosting, evaluating on a random subset of the data{p_end}
{phang2} Load data{p_end}
{phang3} {stata webuse iris, clear}{p_end}
{phang2} Train on about 3/4 of obs{p_end}
{phang3} {stata gen train_flag = runiform()<0.75}{p_end}
{phang2} Run adaptive boosting classifier, training on training sample{p_end}
{phang3} {stata pyada iris seplen sepwid petlen petwid if train_flag==1, type(classify)}{p_end}
{phang2} Alternative syntax for the above: we can use training() and obtain test sample RMSE in one step{p_end}
{phang3} {stata pygradboost iris seplen sepwid petlen petwid, type(classify) training(train_flag)}{p_end}
{phang2} Save predictions in a variable called iris_hat{p_end}
{phang3} {stata predict iris_hat}{p_end}

{pstd}{bf:Example 3}: Regression with gradient boosting{p_end}
{phang2} Load data{p_end}
{phang3} {stata sysuse auto, clear}{p_end}
{phang2} Run adaptive boosting regressor{p_end}
{phang3} {stata pyada price mpg trunk weight, type(regress)}{p_end}
{phang2} Save predictions in a variable called price_hat{p_end}
{phang3} {stata predict price_hat}{p_end}

{pstd}{bf:Example 4}: Regression with gradient boosting, slower learning rate, random subset of data{p_end}
{phang2} Load data{p_end}
{phang3} {stata sysuse auto, clear}{p_end}
{phang2} Train on about 3/4 of obs{p_end}
{phang3} {stata gen train_flag = runiform()<0.75}{p_end}
{phang2} Run adaptive boosted regression, training on training sample{p_end}
{phang3} {stata pygradboost price mpg trunk weight, type(regress) training(train_flag) learning_rate(0.7)}{p_end}
{phang2} Save predictions in a variable called price_hat{p_end}
{phang3} {stata predict price_hat}{p_end}


{marker author}{...}
{title:Author}
 
{pstd}Michael Droste{p_end}
{pstd}mdroste@fas.harvard.edu{p_end}
 
 
{marker acknowledgements}{...}
{title:Acknowledgements}

{pstd}This program owes a lot to the wonderful {browse "https://scikit-learn.org/":scikit-learn} library in Python.


