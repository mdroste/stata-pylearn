*! Version 0.63, 8jul2020, Michael Droste, mdroste@fas.harvard.edu
*! More info and latest version: github.com/mdroste/stata-pylearn
*===============================================================================
* Program:   pygradboost.ado
* Purpose:   Classification and regression in Stata 16+ using gradient boosted
*            trees, with Python and scikit-learn. Component of pylearn.
*===============================================================================

program define pygradboost, eclass
version 16.0
syntax varlist(min=2) [if] [in] [aweight fweight], ///
[ ///
	type(string asis)            	 /// type: classifier or regressor
	n_estimators(integer 50) 	 	 /// number of estimators
	learning_rate(real 0.1)          /// learning rate
	subsample(real 1)                /// frac of sample to use for base learners
	criterion(string asis)           /// algorithm for boosting (classifier)
	loss(string asis)                /// loss function (regressor)
	max_depth(integer 3)	 	 	 /// max tree depth
	min_samples_split(real 2) 	 	 /// min obs before splitting internal node
	min_samples_leaf(real 1) 	 	 /// min obs required at a leaf node
	min_weight_fraction_leaf(real 0) /// min weighted frac of sum of total weights
	max_features(string asis)	 	 /// number of features to consider for best split
	max_leaf_nodes(real -1)	 	 	 /// max leaf nodes
	min_impurity_decrease(real 0)	 /// split if it induces this amt decrease in impurity
	validation_fraction(real 0.1)    /// proportion of training dataset
	n_iter_no_change(integer -1)     /// Terminates when training doesnt improve
	tol(real 0.0001)                 /// Tolerance threshold for early stopping
	ccp_alpha(real 0)                /// complexity parameter
	random_state(integer -1) 	 	 /// seed used by random number generator
	class_weight 			 	 	 /// XX NOT YET IMPLEMENTED
    training(varname) 	             /// training dataset identifier
    standardize                      /// standardize features
]

*-------------------------------------------------------------------------------
* Before doing anything: make sure we have Python 3.0+ and good modules
*-------------------------------------------------------------------------------

pylearn, check

*-------------------------------------------------------------------------------
* Handle arguments
*-------------------------------------------------------------------------------

*-------------------------------------------------
* type: string asis, either classify or regress
*-------------------------------------------------

if "`type'"=="" {
	di as error "ERROR: type() option needs to be specified. Valid options: type(classify) or type(regress)"
	exit 1
}

if "`type'"=="reg" local type "regress"
if "`type'"=="regression" local type "regress"
if ~inlist("`type'","classify","regress") {
	di as error "Syntax error: invalid choice for type (chosen: `type'). Valid options are classify or regress"
	exit 1
}

*-------------------------------------------------
* n_estimators: positive integer (default: 50)
*-------------------------------------------------

if `n_estimators'<1 {
	di as error "Syntax error: Number of estimators must be a positive integer (was `n_estimators')"
	exit 1
}

*-------------------------------------------------
* loss: loss function to use
*-------------------------------------------------

* Exception handling
if "`type'"=="classify" {
	if ~inlist("`loss'","exponential","deviance","") {
		di as error "Syntax error: invalid choice for loss(). You specified loss(`loss')."
		di as error "Since you specified type(classify), valid options are loss(exponential) and loss(deviance)."
		exit 1
	}
}
if "`type'"=="regress" {
	if ~inlist("`loss'","ls", "lad", "huber", "quantile","") {
		di as error "Syntax error: invalid choice for loss(). You specified loss(`loss')."
		di as error "Since you specified type(regress), valid options are loss(ls), loss(lad), loss(huber), and loss(quantile)."
		exit 1
	}
}
if "`type'"=="classify" & "`loss'"=="" local loss deviance
if "`type'"=="regress" & "`loss'"=="" local loss ls

*-------------------------------------------------
* learning_rate: float, default 1
*-------------------------------------------------

* xx

*-------------------------------------------------
* criterion
*-------------------------------------------------

* Exception handling
if ~inlist("`criterion'","friedman_mse","mse","mae","") {
	di as error "Syntax error: invalid choice for criterion(). You specified criterion(`criterion')."
	di as error "Valid options are criterion(friedman_mse), criterion(mse), and criterion(mae)."
	exit 1
}

* Defaults
if "`criterion'"=="" local criterion friedman_mse

*-------------------------------------------------
* max_depth: positive integer (default: None)
*-------------------------------------------------

if "`max_depth'"=="-1" local max_depth None
if "`max_depth'"!="None" {
	if `max_depth'<1 {
		di as error "Syntax error: max_depth() must be positive integer (was `max_depth')"
		exit 1
	}
}

*-------------------------------------------------
* min_samples_split: int, float, optional  (default: 2)
*-------------------------------------------------

if "`min_samples_split'"=="" local min_samples_split 2

*-------------------------------------------------
* min_samples_leaf: int, float, optional (default: 1) 
*-------------------------------------------------

if "`min_samples_leaf'"=="" local min_samples_leaf 1

*-------------------------------------------------
* min_weight_fraction_leaf: float, optional (default: 0)
*-------------------------------------------------

if "`min_weight_fraction_leaf'"=="" local min_weight_fraction_leaf 0

*-------------------------------------------------
* max_features 
* int, float, string, or None, optional (default: "auto")
*-------------------------------------------------

local max_features_di `max_features'
if "`max_features'"=="" {
	local max_features "auto"
	local max_features_di "None"
}
* if not sqrt or log2, then should be float
if ~inlist("`max_features'","auto","sqrt","log2") {
	* check to make sure float
	cap confirm number `max_features'
	if _rc>0 {
		di as error "Syntax error: max_features() should be either 'auto', 'sqrt', 'log2', or an integer/float (was `max_features')"
		exit 1
	}
	if _rc==0 {
		* xx need to apply ceil thing here
	}
}
else {
	local max_features `""`max_features'""'
}

*-------------------------------------------------
* max_leaf_nodes
*-------------------------------------------------

if "`max_leaf_nodes'"=="-1" local max_leaf_nodes None
if "`max_leaf_nodes'"!="None" {
	if `max_leaf_nodes'<1 {
		di as error "Syntax error: if you specify max_leaf_nodes(), make it a positive integer (was `max_leaf_nodes')"
		exit 1
	}
}

*-------------------------------------------------
* min_impurity_decrease: xx test me
*-------------------------------------------------

if "`min_impurity_decrease'"=="" local min_impurity_decrease 0


*-------------------------------------------------
* random_state: initialize random number generator
*-------------------------------------------------

if "`random_state'"=="-1" local random_state None
if "`random_state'"!="" & "`random_state'"!="None" {
	if `random_state'<1 {
		di as error "Syntax error: random_state should be a positive integer."
		exit 1
	}
	set seed `random_state'
}

*-------------------------------------------------
* class_weight: xx not yet implemented
*-------------------------------------------------

if "`class_weight'"=="" local class_weight None

*-------------------------------------------------
* Tolerance
*-------------------------------------------------

* xx

if `n_iter_no_change'==-1 local n_iter_no_change None

*-------------------------------------------------
* training: xx not yet implemented
*-------------------------------------------------


*-------------------------------------------------
* feature importance
*-------------------------------------------------

local importance 0
if "`feature_importance'"!="" local importance 1

*-------------------------------------------------
* standardize
*-------------------------------------------------

if "`standardize'"=="" {
	local standardize 0
	local stdize_fmt "False"
}
else {
	local standardize 1
	local stdize_fmt "True"
}

*-------------------------------------------------------------------------------
* Manipulate data in Stata first
*-------------------------------------------------------------------------------

* Pass varlist into varlists called yvar and xvars
gettoken yvar xvars : varlist
local num_features : word count `xvars'

* Restrict sample with if and in conditions
marksample touse, strok novarlist
tempvar touse2
gen `touse2' = `touse'
ereturn post, esample(`touse2')

* Preserve original data
preserve

* Keep only if/in
qui drop if `touse'==0

* if type(classify): check to see if y needs encoding to numeric
local yvar2 `yvar'
if "`type'"=="classify" {
	capture confirm numeric var `yvar'
	if _rc>0 {
		local needs_encoding "yes"
		encode `yvar', gen(`yvar'_encoded)
		noi di "Encoded `yvar' as `yvar'_encoded"
		local yvar2 `yvar'_encoded
	}
}

* restrict sample to jointly nonmissing observations
foreach v of varlist `varlist' {
	qui drop if mi(`v')
}

* Define a temporary variable for the training sample
local training_di `training'
tempvar training_var
if "`training'"=="" {
	gen `training_var' = 1
	local training_di "None"
}
if "`training'"!="" gen `training_var' = `training'

* Get number of obs in train and validate samples
qui count if `training_var'==1
local num_obs_train = `r(N)'
qui count
local num_obs_test = `r(N)' - `num_obs_train'
local nonempty_test = `num_obs_test'>0

*-------------------------------------------------------------------------------
* Call Python
*-------------------------------------------------------------------------------

* Store a macro to slightly change results table
if "`type'"=="regress" local type_str "regression"
if "`type'"=="classify" local type_str "classification"

* Pass options to Python to import data, run random forest regression, return results
python: run_gradboost( ///
	"`type'", ///
	"`training_var' `yvar2' `xvars'", "`training_var'", ///
	`n_estimators', "`loss'", `learning_rate', "`criterion'", ///
	`max_depth', `min_samples_split', `min_samples_leaf', `min_weight_fraction_leaf', ///
	`max_features', `max_leaf_nodes', `min_impurity_decrease', `subsample', ///
	`random_state', `class_weight', `standardize', `validation_fraction', `n_iter_no_change', `tol', `ccp_alpha')

*-------------------------------------------------------------------------------
* Prep data to prepare for output
*-------------------------------------------------------------------------------
	
* Format random number seed
if "`random_state'"=="None" local seed_di "None"
if "`random_state'"!="None" local seed_di `random_state'

* Format RMSE/MAE
local is_rmse: di %10.4f `e(training_rmse)'
local is_mae: di %10.4f `e(training_mae)'
local os_rmse: di %10.4f `e(test_rmse)'
local os_mae: di %10.4f `e(test_mae)'
local train_obs_f: di %10.0fc `num_obs_train'
local test_obs_f: di %10.0fc `num_obs_test'

* Format/truncate dependent variable name
local yvarlen = length("`yvar'")
local yvar_fmt = "`yvar'"
if `yvarlen'>13 {
	local yvar_fmt = substr("`yvar'",1,13) + "..."
}

*-------------------------------------------------------------------------------
* Display output in terminal
*-------------------------------------------------------------------------------

* Display output
noi di "{hline 80}"
noi di in ye "Gradient boosted `type_str'"
noi di " "
noi di in gr "{ul:Data}"
noi di in gr "Dependent variable    = " in ye "`yvar_fmt'" _continue
noi di in gr _col(41) "Number of training obs   =" _continue
noi di in ye "`train_obs_f'"
noi di in gr "Number of features    = " in ye `num_features' _continue
noi di in gr _col(41) "Number of validation obs =" _continue
noi di in ye "`test_obs_f'"
noi di in gr "Training identifier   = " in ye "`training_di'"
noi di " "
noi di in gr "{ul:Options}"
noi di in gr "Number of estimators  = " in ye "`n_estimators'" _continue
noi di in gr _col(41)  "Loss function            = " in ye "`loss'"
noi di in gr "Learning rate         = " in ye "`learning_rate'" _continue
noi di in gr _col(41) "Split criterion          = " in ye "`criterion'"
noi di in gr "Min impurity decrease = " in ye "`min_impurity_decrease'" _continue
noi di in gr _col(41) "Min obs/interior node    = " in ye "`min_samples_split'"
noi di in gr "Max features/tree     = " in ye "`max_features_di'" _continue
noi di in gr _col(41) "Min obs/leaf             = " in ye "`min_samples_leaf'"
noi di in gr "Max leaf nodes        = " in ye "`max_leaf_nodes'" _continue
noi di in gr _col(41) "Min weight fraction/leaf = " in ye "`min_weight_fraction_leaf'"
noi di in gr "Random number seed    = " in ye "`seed_di'"
noi di " "
noi di in gr "{ul:Output}"
if "`type'"=="regress"  noi di in gr "Training RMSE       = " in ye `is_rmse'
if "`type'"=="classify" noi di in gr "Training accuracy   = " in ye `e(training_accuracy)'
if "`type'"=="regress"  & `nonempty_test'==1 noi di in gr "Validation RMSE     = " in ye `os_rmse'
if "`type'"=="classify" & `nonempty_test'==1 noi di in gr "Validation accuracy = " in ye `e(test_accuracy)'

noi di " "
noi di in gr "Type {help pygradboost:help pygradboost} to access the documentation."
noi di "{hline 80}"

*-------------------------------------------------------------------------------
* Return stuff to e class
*-------------------------------------------------------------------------------

* Count features so I can return it
local K = 0
foreach v of varlist `xvars' {
	local K = `K'+1
}

* Ereturn scalars
ereturn scalar N = `num_obs_train'
ereturn scalar N_test = `num_obs_test'
ereturn scalar K = `num_features'

* Ereturn locals
ereturn local predict "pylearn_predict"
ereturn local features "`xvars'"
ereturn local depvar "`yvar'"
ereturn local trainflag "`training'"
ereturn local cmd "pytree"
ereturn local type "`type'"


end

*===============================================================================
* Python helper functions
*===============================================================================

version 16.0
python:

#-------------------------------------------------------------------------------
# Import required packages and attempt to install w/ Pip if that fails
#-------------------------------------------------------------------------------

# Import required Python modules (pandas, scikit-learn, sfi)
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sfi import Data,Matrix,Scalar
from sklearn import metrics
import numpy as np

# To pass objects to Stata
import __main__

# Set random seed
import random
random.seed(50)

#-------------------------------------------------------------------------------
# Define Python function: run_gradboost
#-------------------------------------------------------------------------------

def run_gradboost(type,vars,training,n_estimators, loss, learning_rate, criterion, 
                 max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, 
                 max_features, max_leaf_nodes, min_impurity_decrease, 
                 subsample, random_state, class_weight, standardize, 
                 validation_fraction, n_iter_no_change, tol, ccp_alpha):

	#-------------------------------------------------------------
	# Load data from Stata into Python
	#-------------------------------------------------------------
	
	# Load into Pandas data frame
	df = DataFrame(Data.get(vars))
	colnames = []
	for var in vars.split():
		 colnames.append(var)
	df.columns = colnames

	# Split training data and test data into separate data frames
	df_train, df_test = df[df[training]==1], df[df[training]==0]

	# Create list of feature names
	features = df.columns[2:]
	y        = df.columns[1]

	# Split training data frame into features (x) and outcome (y)
	x_insample  = df_train[features]
	x           = df[features]
	y_insample  = df_train[y]

	#-------------------------------------------------------------
	# Preprocessing: if standardizing, scale data
	#-------------------------------------------------------------

	# If standardizing, scale features to mean zero, std dev one
	if standardize==1:
		scaler = preprocessing.StandardScaler().fit(x_insample)
		x_insample = scaler.transform(x_insample)
		x_insample = DataFrame(x_insample)
		x_insample.columns = features
		x = scaler.transform(x)
		x = DataFrame(x)
		x.columns = features
	
	#-------------------------------------------------------------
	# Initialize model objects
	#-------------------------------------------------------------

    # Initialize regressor (if model type is regress)
	if type=="regress":
		model = GradientBoostingRegressor(n_estimators=n_estimators, 
										 learning_rate=learning_rate, 
										 subsample=subsample, 
										 criterion=criterion, 
										 min_samples_split=min_samples_split, 
										 min_samples_leaf=min_samples_leaf, 
										 min_weight_fraction_leaf=min_weight_fraction_leaf, 
										 max_depth=max_depth, 
										 min_impurity_decrease=min_impurity_decrease, 
										 random_state=random_state, 
										 max_features=max_features, 
										 max_leaf_nodes=max_leaf_nodes, 
										 validation_fraction=validation_fraction, 
										 n_iter_no_change=n_iter_no_change, 
										 tol=tol, 
										 ccp_alpha=ccp_alpha)

	# Initialize classifier (if model type is classify)
	if type=="classify":
		model = GradientBoostingClassifier(n_estimators=n_estimators, 
										 learning_rate=learning_rate, 
										 subsample=subsample, 
										 criterion=criterion, 
										 min_samples_split=min_samples_split, 
										 min_samples_leaf=min_samples_leaf, 
										 min_weight_fraction_leaf=min_weight_fraction_leaf, 
										 max_depth=max_depth, 
										 min_impurity_decrease=min_impurity_decrease, 
										 random_state=random_state, 
										 max_features=max_features, 
										 max_leaf_nodes=max_leaf_nodes, 
										 validation_fraction=validation_fraction, 
										 n_iter_no_change=n_iter_no_change, 
										 tol=tol, 
										 ccp_alpha=ccp_alpha)

					 
	#-------------------------------------------------------------
	# Fit model, get predictions, pass objects back to main
	#-------------------------------------------------------------

	# Train model on training data
	model.fit(x_insample, y_insample)

	# Get in-sample prediction
	pred_insample = model.predict(x_insample)

	# Get full-sample prediction
	model_predict = model.predict(x)

	# Pass objects back to __main__ namespace to interact w/ later
	__main__.model_object = model
	__main__.model_predict = model_predict
	__main__.features = features

	#-------------------------------------------------------------
	# Get model diagnostics (RMSE, classification accuracy, etc)
	#-------------------------------------------------------------

	# If regression: get training sample rmse/mae
	if type=="regress":
		insample_mae = metrics.mean_absolute_error(y_insample, pred_insample)
		insample_mse = metrics.mean_squared_error(y_insample, pred_insample)
		insample_rmse = np.sqrt(insample_mse)
		Scalar.setValue("e(training_mae)", insample_mae, vtype='visible')
		Scalar.setValue("e(training_rmse)", insample_rmse, vtype='visible')
	
	# If classify: get training sample accuracy
	if type=="classify":
		insample_accuracy = metrics.accuracy_score(y_insample, pred_insample)
		Scalar.setValue("e(training_accuracy)", insample_accuracy, vtype='visible')
		
	#-----------------------------------
	# Get test sample fit (if applicable)
	#-----------------------------------

	# If nonempty test
	if len(df_test)!=0:

		# Generate predictions
		pred_outsample = model.predict(df_test[features])
		y_outsample = df_test[y]

		# If regression: get training sample rmse/mae
		if type=="regress":
			outsample_mae = metrics.mean_absolute_error(y_outsample, pred_outsample)
			outsample_mse = metrics.mean_squared_error(y_outsample, pred_outsample)
			outsample_rmse = np.sqrt(outsample_mse)
			Scalar.setValue("e(test_mae)", outsample_mae, vtype='visible')
			Scalar.setValue("e(test_rmse)", outsample_rmse, vtype='visible')

		# If classify: get training sample accuracy
		if type=="classify":
			outsample_accuracy = metrics.accuracy_score(y_outsample, pred_outsample)
			Scalar.setValue("e(test_accuracy)", outsample_accuracy, vtype='visible')


end
