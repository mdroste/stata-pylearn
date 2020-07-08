{smcl}
{* *! version 0.63 8jul2020}{...}
{viewerjumpto "Syntax" "pytree##syntax"}{...}
{viewerjumpto "Description" "pytree##description"}{...}
{viewerjumpto "Options" "pytree##options"}{...}
{viewerjumpto "Stored results" "pyforest##results"}{...}
{viewerjumpto "Examples" "pytree##examples"}{...}
{viewerjumpto "Author" "pytree##author"}{...}
{viewerjumpto "Acknowledgements" "pytree##acknowledgements"}{...}
{title:Title}
 
{p2colset 5 15 21 2}{...}
{p2col :{hi:pytree} {hline 2}}Decision tree regression and classification with Python and scikit-learn{p_end}
{p2colreset}{...}
 
 
{marker syntax}{title:Syntax}
 
{p 8 15 2}
{cmd:pytree} {depvar} {indepvars} {ifin}, type(string) [{cmd:}{it:options}]
                               
 
{synoptset 32 tabbed}{...}
{synopthdr :options}
{synoptline}
 
{syntab :Main}
{synopt :{opt type(string)}}{it:string} may be {bf:regress} or {bf:classify}.{p_end}
 
{syntab :Decision tree options}
{synopt :{opt criterion(string)}}Criterion for splitting nodes (see details below){p_end}
{synopt :{opt max_depth(#)}}Maximum tree depth{p_end}
{synopt :{opt min_samples_split(#)}}Minimum observations per node{p_end}
{synopt :{opt min_weight_fraction_leaf(#)}}Min fraction at leaf{p_end}
{synopt :{opt max_features(numeric)}}Maximum number of features to consider per tree{p_end}
{synopt :{opt max_leaf_nodes(#)}}Maximum leaf nodes{p_end}
{synopt :{opt min_impurity_decrease(#)}}Propensity to split{p_end}

{syntab :Training options}
{synopt :{opt training(varname)}}varname is an indicator for the training sample{p_end}

{synoptline}
{p 4 6 2}
{p_end}
 
 
{marker description}{...}
{title:Description}
 
{pstd}
{opt pytree} performs regression or classification with decision trees. 

{pstd} In particular, {opt pytree} implements decision trees using Python's scikit-learn module; specifically, the decisionTreeClassifier decisionTreeRegression methods.

{pstd} Note that {opt pytree} relies on the Python integration functionality introduced with Stata 16. Therefore, users will need Stata 16, Python (preferably 3.x), and the scikit-learn library installed in order to run this ado-file.

{pstd} For more information on pytree, refer to the pylearn GitHub page: {browse "https:/www.github.com/mdroste/stata-pylearn":https:/www.github.com/mdroste/stata-pylearn}.

 
{marker options}{...}
{title:Options}
 
{dlgtab:Main}
 
{phang}
{opth type(string)} declares whether this is a regression or classification problem. In general, type(classify) is more appropriate when the dependent variable is categorical, and type(regression) is more appropriate when the dependent variable is continuous.
 

{dlgtab:Training data options}

{phang}
{opt training(varname)} identifies an indicator variable in the current dataset that is equal to 1 when an observation should be used for training and 0 otherwise.


{dlgtab:Decision tree options}
 
{phang}

{phang}
{opt criterion(string)} determines the function used to measure the quality of a proposed split. Valid options for criterion() depend on whether the task is a classification task or a regression task. If type(regress) is specified, valid options are mse (default) and mae. If type(classify) is specified, valid options are gini (default) and entropy. 

{phang}
{opt max_depth(#)} specifies the maximum tree depth. By default, this is None.

{phang}
{opt min_samples_split(#)} specifies the minimum number of observations required to consider splitting an internal node of a tree. By default, this is 2.

{phang}
{opt min_samples_leaf(#)} specifies the minimum number of observations required at each 'leaf' node of a tree. By default, this is 1.

{phang}
{opt min_weight_fraction_leaf(#)} specifies the minimum weighted fraction of the sum of weights required at each leaf node. When weights are not specified, this is simply the minimum fraction of observations required at each leaf node. By default, this is 0.

{phang}
{opt max_features(string)} specifies the number of features to consider when looking for the best split. By default, this is equal to the number of features (aka independent variables). Other options are max_features(sqrt) (the square root of the number of features), max_features(log2) (the base-2 logarithm of the number of features), an integer, or a float. If a non-integer is specified, then int(max_features,number of features) are considered at each split.

{phang}
{opt max_leaf_nodes(#)} Grow trees with max_leaf_nodes in best-first fashion, where best is defined in terms of relative reduction in impurity. By default, an unlimited number of leaf nodes. 

{phang}
{opt min_impurity_decrease(#)} determines the threshold such tha a node is split if it induces a decrease in impurity criterion greater than or equal to this value. By default, this is 0. 


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

{synoptset 24 tabbed}{...}
{syntab:Matrices}
{synopt:{cmd:e(importance)}}Variable importance scores{p_end}


{marker examples}{...}
{title:Examples}
 
{pstd}See the Github page.{p_end}

{pstd}{bf:Example 1}: Classification with decision tree, saving predictions as a new variable called iris_prediction{p_end}
{phang2} Load data{p_end}
{phang3} {stata webuse iris, clear}{p_end}
{phang2} Run decision tree classifier{p_end}
{phang3} {stata pytree iris seplen sepwid petlen petwid, type(classify)}{p_end}
{phang2} Save predictions in a variable called iris_hat{p_end}
{phang3} {stata predict iris_hat}{p_end}

{pstd}{bf:Example 2}: Classification with decision trees, evaluating on a random subset of the data{p_end}
{phang2} Load data{p_end}
{phang3} {stata webuse iris, clear}{p_end}
{phang2} Train on about 3/4 of obs{p_end}
{phang3} {stata gen train_flag = runiform()<0.75}{p_end}
{phang2} Run decision tree classifier, training on training sample{p_end}
{phang3} {stata pytree iris seplen sepwid petlen petwid if train_flag==1, type(classify)}{p_end}
{phang2} Alternative syntax for the above: we can use training() and obtain test sample RMSE in one step{p_end}
{phang3} {stata pytree iris seplen sepwid petlen petwid, type(classify) training(train_flag)}{p_end}
{phang2} Save predictions in a variable called iris_hat{p_end}
{phang3} {stata predict iris_hat2}{p_end}

{pstd}{bf:Example 3}: Decision tree regression{p_end}
{phang2} Load data{p_end}
{phang3} {stata sysuse auto, clear}{p_end}
{phang2} Run decision tree regression model{p_end}
{phang3} {stata pytree price mpg trunk weight, type(regress)}{p_end}
{phang2} Save predictions in a variable called price_hat{p_end}
{phang3} {stata predict price_hat}{p_end}

{pstd}{bf:Example 4}: Decision tree regression, limiting tree depth, random subset of data{p_end}
{phang2} Load data{p_end}
{phang3} {stata sysuse auto, clear}{p_end}
{phang2} Train on about 3/4 of obs{p_end}
{phang3} {stata gen train_flag = runiform()<0.75}{p_end}
{phang2} Run decision tree regression model, training on training sample{p_end}
{phang3} {stata pytree price mpg trunk weight, type(regress) max_depth(2) training(train_flag)}{p_end}
{phang2} Save predictions in a variable called price_hat{p_end}
{phang3} {stata predict price_hat}{p_end}

 
{marker author}{...}
{title:Author}
 
{pstd}Michael Droste{p_end}
{pstd}mdroste@fas.harvard.edu{p_end}
 
 
{marker acknowledgements}{...}
{title:Acknowledgements}

{pstd}This program owes a lot to the wonderful {browse "https://scikit-learn.org/":scikit-learn} library in Python.


