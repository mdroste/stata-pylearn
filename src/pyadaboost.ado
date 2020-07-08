*! Version 0.63, 8jul2020, Michael Droste, mdroste@fas.harvard.edu
*! More info and latest version: github.com/mdroste/stata-pylearn
*===============================================================================
* Program:   pyadaboost.ado
* Purpose:   Adaptive boosting classification and regression in Stata 16+ with
*            Python and scikit-learn. Component of pylearn.
*===============================================================================

program define pyadaboost, eclass
version 16.0
syntax varlist(min=2) [if] [in] [aweight fweight], ///
[ ///
	type(string asis)            	 /// random forest type: classifier or regressor
	n_estimators(integer 50) 	 	 /// number of trees
	learning_rate(real 1)           /// learning rate
	algorithm(string asis)           /// algorithm for boosting (classifier)
	loss(string asis)                /// loss function (regressor)
	random_state(integer -1) 	 	 /// seed used by random number generator
	class_weight 			 	 	 /// XX NOT YET IMPLEMENTED
    training(varname) 	             /// training dataset identifier
    standardize                      /// standardize variables to unit variance, mean zero
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
* learning_rate: float, defualt 1
*-------------------------------------------------


*-------------------------------------------------
* Algorithm
*-------------------------------------------------

* Send a warning if nonempty and regressor specified
if "`algorithm'"!="" & "`type'"=="regress" {
	di "Warning: algorithm(`algorithm') and type(regress) specified."
	di "Note that algorithm(`algorithm') only applies for classification & will be ignored."
}

* Set default to samme.r if not specified
if "`algorithm'"=="" local algorithm "SAMME.R"

* take lowercase to uppercase
if "`algorithm'"=="samme" local algorithm "SAMME"
if "`algorithm'"=="samme.r" local algorithm "SAMME.R"

* Throw error if invalid type
if ~inlist("`algorithm'","SAMME","SAMME.R") {
	di as error "Error: Invalid option specified for algorithm()."
	di as error "Valid options for algorithm() are SAMME and SAMME.R"
	exit 1
}

*-------------------------------------------------
* Loss function
*-------------------------------------------------

* Send a warning if nonempty and classifier specified
if "`loss'"!="" & "`type'"=="classify" {
	di "Warning: loss(`loss') and type(classify) specified."
	di "Note that loss(`loss') only applies for regression & will be ignored."
}

* Set default to linear loss fn if not specified
if "`loss'"=="" local loss "linear"

* Throw error if invalid type
if ~inlist("`loss'","linear","square","exponential") {
	di as error "Error: Invalid option specified for loss()."
	di as error "Valid options for loss() are linear, square, and exponential"
	exit 1
}

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
python: run_adaboost( ///
	"`type'", ///
	"`training_var' `yvar2' `xvars'", "`training_var'", ///
	`n_estimators', `learning_rate', "`algorithm'", ///
	"`loss'", `random_state', `class_weight', `standardize')


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
noi di in ye "Adaptive boosted `type_str'"
noi di " "
noi di in gr "{ul:Data}"
noi di in gr "Dependent variable  = " in ye "`yvar_fmt'" _continue
noi di in gr _col(41) "Number of training obs   =" _continue
noi di in ye "`train_obs_f'"
noi di in gr "Number of features  = " in ye `num_features' _continue
noi di in gr _col(41) "Number of validation obs =" _continue
noi di in ye "`test_obs_f'"
noi di in gr "Training identifier = " in ye "`training_di'"
noi di " "
noi di in gr "{ul:Options}"
noi di in gr "Number of estimators = " in ye "`n_estimators'" _continue
if "`type'"=="regress"  noi di in gr _col(41) "Loss function            = " in ye "`loss'"
if "`type'"=="classify" noi di in gr _col(41) "Boosting algorithm       = " in ye "`algorithm'"
noi di in gr "Learning rate        = " in ye "`learning_rate'" _continue
noi di in gr _col(41) "Random number seed       = " in ye "`seed_di'"
noi di " "
noi di in gr "{ul:Output}"
if "`type'"=="regress"  noi di in gr "Training RMSE       = " in ye `is_rmse'
if "`type'"=="classify" noi di in gr "Training accuracy   = " in ye `e(training_accuracy)'
if "`type'"=="regress"  & `nonempty_test'==1 noi di in gr "Validation RMSE     = " in ye `os_rmse'
if "`type'"=="classify" & `nonempty_test'==1 noi di in gr "Validation accuracy = " in ye `e(test_accuracy)'

noi di " "
noi di in gr "Type {help pyadaboost:help pyadaboost} to access the documentation."
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
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sfi import Data,Matrix,Scalar
from sklearn import metrics, preprocessing
import numpy as np


# To pass objects to Stata
import __main__

# Set random seed
import random

#-------------------------------------------------------------------------------
# Define Python function: run_adaboost
#-------------------------------------------------------------------------------

def run_adaboost(type,vars,training,n_estimators,learning_rate,algorithm,loss,random_state,class_weight, standardize):

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
		model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state=random_state)

	# Initialize classifier (if model type is classify)
	if type=="classify":
		model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm, random_state=random_state)
	
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
