{smcl}
{* *! version 0.65 8jul2020}{...}
{viewerjumpto "Syntax" "pyadaboost##syntax"}{...}
{viewerjumpto "Description" "pyadaboost##description"}{...}
{viewerjumpto "Options" "pyadaboost##options"}{...}
{viewerjumpto "Stored results" "pyadaboost##results"}{...}
{viewerjumpto "Examples" "pyadaboost##examples"}{...}
{viewerjumpto "Author" "pyadaboost##author"}{...}
{viewerjumpto "Acknowledgements" "pyadaboost##acknowledgements"}{...}
{title:Title}
 
{p2colset 5 14 21 2}{...}
{p2col :{hi:pyadaboost} {hline 2}}Adaptive Boosting with Python and scikit-learn{p_end}
{p2colreset}{...}
 
{marker syntax}{title:Syntax}
 
{p 4 15 2}
{cmd:pyadaboost} {depvar} {indepvars} {ifin}, type(string) [{cmd:}{it:options}]
                               
{synoptset 32 tabbed}{...}
{synopthdr :options}
{synoptline}
{syntab :Main}
{synopt :{opt type(string)}}{it:string} may be {bf:regress} or {bf:classify}.{p_end}

{syntab :Pre-processing}
{synopt :{opt training(varname)}}varname is an indicator for the training sample{p_end}

{syntab :Adaptive Boosting options}
{synopt :{opt n_estimators(#)}}Number of trees{p_end}
{synopt :{opt learning_rate(#)}}Shrinks the contribution of each predictor{p_end}
{synopt :{opt loss(string)}}Loss function when updating weights, for type(regress){p_end}
{synopt :{opt algorithm(string)}}Boosting algorithm, for type(classify){p_end}

{synoptline}
{p 4 6 2}
{p_end}
    For more information on syntax and options, see the {browse "https://scikit-learn.org/stable/modules/ensemble.html#adaboost":scikit-learn documentation for AdaBoost}.
    {browse "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html":AdaBoost regression syntax}
    {browse "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html":AdaBoost classification syntax}
 
{marker description}{...}
{title:Description}
 
{pstd}
{opt pyadaboost} performs regression or classification with the adaptive boosting algorithm introduced by Freund and Schapire (1995). AdaBoost is a meta-estimator that begins by fitting a regression/classification problem on the original dataset, and then on a sequence of successive datasets where higher weight is given to observations with larger prediction errors from the previous iteration.

{pstd} In particular, {opt pyadaboost} provides a wrapper around the AdaBoost implementations in scikit-learn. These implementations involve three algorithms: {p_end}
{p2col 8 12 12 2: a)}The AdaBoost-SAMME and AdaBoost-SAMME.R algorithms for classification (Zhu, Zou, Rosset, and Hastie 2009){p_end}
{p2col 8 12 12 2: b)}AdaBoost.R2 algorithm for regression (Drucker 1997).{p_end}

{pstd} {opt pyadaboost} relies critically on the Python integration functionality introduced with Stata 16. Therefore, users will need Stata 16, Python 3.6+, and the scikit-learn library installed in order to run.

{pstd} For more information on pyadaboost, refer to the pylearn GitHub page: {browse "https:/www.github.com/mdroste/stata-pylearn":https:/www.github.com/mdroste/stata-pylearn}.


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
{opt n_estimators(#)} determines the number of estimators at which boosting is terminated. 

{phang}
{opt learning_rate(#)} shrinks the contribution of each predictor. There is a tradeoff between learning_rate() and n_estimators(). The default is 1. 

{phang}
{opt algorithm(string)} specifies the real boosting algorithm. This option is only used when type(classify) is chosen. The default is 'SAMME.R'. The alternative option is algorithm('SAMME'), a discrete boosting algorithm.

{phang}
{opt loss(string)} specifies the loss function used when type(regress) is specified. The default is loss(linear). Alternative options are loss(exponential) and loss(square). 


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

{pstd}{bf:Example 1}: Classification with adaptive boosting, saving predictions as a new variable called iris_prediction{p_end}
{phang2} Load data{p_end}
{phang3} {stata webuse iris, clear}{p_end}
{phang2} Run adaptive boosting classifier{p_end}
{phang3} {stata pyadaboost iris seplen sepwid petlen petwid, type(classify)}{p_end}
{phang2} Save predictions in a variable called iris_hat{p_end}
{phang3} {stata predict iris_hat}{p_end}

{pstd}{bf:Example 2}: Classification with adaptive boosting, evaluating on a random subset of the data{p_end}
{phang2} Load data{p_end}
{phang3} {stata webuse iris, clear}{p_end}
{phang2} Train on about 3/4 of obs{p_end}
{phang3} {stata gen train_flag = runiform()<0.75}{p_end}
{phang2} Run adaptive boosting classifier, training on training sample{p_end}
{phang3} {stata pyadaboost iris seplen sepwid petlen petwid if train_flag==1, type(classify)}{p_end}
{phang2} Alternative syntax for the above: we can use training() and obtain test sample RMSE in one step{p_end}
{phang3} {stata pyadaboost iris seplen sepwid petlen petwid, type(classify) training(train_flag)}{p_end}
{phang2} Save predictions in a variable called iris_hat{p_end}
{phang3} {stata predict iris_hat}{p_end}

{pstd}{bf:Example 3}: Regression with adaptive boosting{p_end}
{phang2} Load data{p_end}
{phang3} {stata sysuse auto, clear}{p_end}
{phang2} Run adaptive boosting regressor{p_end}
{phang3} {stata pyadaboost price mpg trunk weight, type(regress)}{p_end}
{phang2} Save predictions in a variable called price_hat{p_end}
{phang3} {stata predict price_hat}{p_end}

{pstd}{bf:Example 4}: Regression with adaptive boosting, slower learning rate, random subset of data{p_end}
{phang2} Load data{p_end}
{phang3} {stata sysuse auto, clear}{p_end}
{phang2} Train on about 3/4 of obs{p_end}
{phang3} {stata gen train_flag = runiform()<0.75}{p_end}
{phang2} Run adaptive boosted regression, training on training sample{p_end}
{phang3} {stata pyadaboost price mpg trunk weight, type(regress) training(train_flag) learning_rate(0.7)}{p_end}
{phang2} Save predictions in a variable called price_hat{p_end}
{phang3} {stata predict price_hat}{p_end}

 
{marker author}{...}
{title:Author}
 
{pstd}Michael Droste{p_end}
{pstd}mdroste@fas.harvard.edu{p_end}
 
 
 
{marker acknowledgements}{...}
{title:Acknowledgements}

{pstd}This program owes a lot to the wonderful {browse "https://scikit-learn.org/":scikit-learn} library in Python.


