*! Version 0.63, 8jul2020, Michael Droste, mdroste@fas.harvard.edu
*! More info and latest version: github.com/mdroste/stata-pylearn
*===============================================================================
* Program:   pyforest.ado
* Purpose:   Random forest classification and regression in Stata 16+ with
*            Python and scikit-learn. Component of pylearn.
*===============================================================================

program define pyforest, eclass
version 16.0
syntax varlist(min=2) [if] [in] [aweight fweight], ///
[ ///
	type(string asis)            	 /// model type: classifier or regressor
	n_estimators(integer 100) 	 	 /// number of trees
	criterion(string asis) 	 	 	 /// split criterion (gini, entropy)
	max_depth(integer -1)	 	 	 /// max tree depth
	min_samples_split(real 2) 	 	 /// min obs before splitting internal node
	min_samples_leaf(real 1) 	 	 /// min obs required at a leaf node
	min_weight_fraction_leaf(real 0) /// min weighted frac of sum of total weights
	max_features(string asis)	 	 /// number of features to consider for best split
	max_leaf_nodes(real -1)	 	 	 /// max leaf nodes
	min_impurity_decrease(real 0)	 /// split if it induces this amt decrease in impurity
	nobootstrap 		 		 	 /// use bootstrap or not
	oob_score 		 		 	 	 /// whether to use out-of-bag obs to estimate generalization accuracy
	n_jobs(integer -1)		 	 	 /// number of processors to use when computing stuff - default is all
	random_state(integer -1) 	 	 /// seed used by random number generator
	verbose		 			 	 	 /// controls verbosity
	warm_start(string asis)	 	 	 /// when set to true, reuse solution of previous call to fit
	class_weight 			 	 	 /// XX NOT YET IMPLEMENTED
    frac_training(real 1)	 	 	 /// randomly assign fraction X to training
    training(varname) 	             /// training dataset identifier
	prediction(string asis) 	     /// variable name to save predictions
	standardize                  	 /// standardize features
]

*-------------------------------------------------------------------------------
* Before doing anything: make sure we have Python 3.0+ and good modules
*-------------------------------------------------------------------------------

pylearn, check

*-------------------------------------------------------------------------------
* Handle arguments
*-------------------------------------------------------------------------------

*-------------------------------------------------
* n_estimators: positive integer (default: 10)
*-------------------------------------------------

if `n_estimators'<1 {
	di as error "Syntax error: Number of trees must be a positive integer (was `n_estimators')"
	exit 1
}

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
* criterion option
*-------------------------------------------------

if "`type'"=="classify" & "`criterion'"=="" local criterion "gini"
if "`type'"=="regress"  & "`criterion'"=="" local criterion "mse"
if "`type'"=="classify" {
	if ~inlist("`criterion'","gini","entropy") {
		di as error "Syntax error: with type(`type'), criterion() must be 'gini' or 'entropy' (was `criterion')"
		exit 1
	}
}
if "`type"=="regress" {
	if ~inlist("`criterion'","mse","mae") {
		di as error "Syntax error: with type(`type'), criterion() must be 'mse' or 'mae' (was `criterion')"
		exit 1
	}
}

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
* max_leaf_nodes: xx test me
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
* nobootstrap: whether or not to bootstrap samples in each tree
*-------------------------------------------------

if "`nobootstrap'"=="" local bootstrap True
if "`nobootstrap'"!="" local bootstrap False

*-------------------------------------------------
* oob_score: xx not yet implemented
*-------------------------------------------------

if "`oob_score'"=="" local oob_score False

*-------------------------------------------------
* n_jobs: number of processors to use in computing random forests
if "`n_jobs'"=="" local n_jobs -1
if `n_jobs'<1 & `n_jobs'!=-1 {
	di as error "Syntax error: num_jobs() must be positive integer or -1."
	di as error " num_jobs() specifies number of processors to use; the default -1 means all."
	di as error " If not -1, this has to be a positive integer. But you should probably not mess around with this."
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
* verbose: control verbosity of python output
*-------------------------------------------------

if "`verbose'"=="" local verbose 0

*-------------------------------------------------
* warm_start: Unsupported scikit-learn option used to use pre-existing rf object 
*-------------------------------------------------

if "`warm_start'"=="" local warm_start False

*-------------------------------------------------
* class_weight: xx not yet implemented
*-------------------------------------------------

if "`class_weight'"=="" local class_weight None

*-------------------------------------------------
* prediction: cant already be a variable name
*-------------------------------------------------

local nopredict = 0
if "`predict'"=="" {
	local nopredict = 1
    
}
capture confirm new variable `prediction'
if _rc>7 {
	di as error "Error: prediction() cannot specify an existing variable (`prediction' already exists)"
	exit 1
}

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
* Manipulate data in Stata
*-------------------------------------------------------------------------------

* Pass varlist into varlists called yvar and xvars
gettoken yvar xvars : varlist
local num_features : word count `xvars'

* Generate an index of original data so we can easily merge back on the results
* xx there is probably a better way to do this... feels inefficient
* xx only needs to be done if saving predictions
tempvar index
gen `index' = _n

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

* Restrict sample to jointly nonmissing observations
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
* Format strings and display text box
*-------------------------------------------------------------------------------

* Store a macro to slightly change results table
if "`type'"=="regress" local type_str "regression"
if "`type'"=="classify" local type_str "classification"

* Pass options to Python to import data, run random forest regression, return results
python: run_random_forest( ///
	"`type'", ///
	"`training_var' `yvar2' `xvars'", ///
	`n_estimators', ///
	"`criterion'", ///
	`max_depth', ///
	`min_samples_split', ///
	`min_samples_leaf', ///
	`min_weight_fraction_leaf', ///
	`max_features', ///
	`max_leaf_nodes', ///
	`min_impurity_decrease', ///
	`bootstrap', ///
	`oob_score', ///
	`n_jobs', ///
	`random_state', ///
	`verbose', ///
	`warm_start', ///
	`class_weight', ///
	"`prediction'", ///
	"`training_var'", ///
	`importance', `nonempty_test', `standardize')
	
* xx move me
if "`prediction'"=="" local prediction_di "Not specified. Use {help predict:predict} for post-estimation predictions."
if "`prediction'"!="" local prediction_di "`prediction'"
if "`random_state'"=="None" local seed_di "None"
if "`random_state'"!="None" local seed_di `random_state'

* xx move me 2
local is_rmse: di %10.4f `e(training_rmse)'
local is_mae: di %10.4f `e(training_mae)'
local os_rmse: di %10.4f `e(test_rmse)'
local os_mae: di %10.4f `e(test_mae)'
local train_obs_f: di %10.0fc `num_obs_train'
local test_obs_f: di %10.0fc `num_obs_test'

* xx move me 3: truncate dependent var name
local yvarlen = length("`yvar'")
local yvar_fmt = "`yvar'"
if `yvarlen'>13 {
	local yvar_fmt = substr("`yvar'",1,13) + "..."
}

* Display output
noi di "{hline 80}"
noi di in ye "Random forest `type_str'"
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
noi di in gr "Number of trees     = " in ye "`n_estimators'" 
noi di in gr "Max tree depth      = " in ye "`max_depth'" _continue
noi di in gr _col(41) "Min obs/leaf             = " in ye "`min_samples_leaf'"
noi di in gr "Max features/tree   = " in ye "`max_features_di'" _continue
noi di in gr _col(41) "Min obs/interior node    = " in ye "`min_samples_split'"
noi di in gr "Max leaf nodes      = " in ye "`max_leaf_nodes'" _continue
noi di in gr _col(41) "Min weight fraction/leaf = " in ye "`min_weight_fraction_leaf'"
noi di in gr "Split criterion     = " in ye "`criterion'" _continue
noi di in gr _col(41) "Min impurity decrease    = " in ye "`min_impurity_decrease'"
noi di in gr "Random number seed  = " in ye "`seed_di'"
noi di " "
noi di in gr "{ul:Output}"
if "`type'"=="regress" {
	noi di in gr "Training RMSE       = " in ye `is_rmse'
}
if "`type'"=="classify" {
	noi di in gr "Training accuracy   = " in ye `e(training_accuracy)'
}
if "`type'"=="regress" & `nonempty_test'==1 {
	noi di in gr "Validation RMSE     = " in ye `os_rmse'
}
if "`type'"=="classify" & `nonempty_test'==1 {
	noi di in gr "Validation accuracy = " in ye `e(test_accuracy)'
}
noi di " "
noi di in gr "Type {help pyforest:help pyforest} to access the documentation."
noi di "{hline 80}"

	
*-------------------------------------------------------------------------------
* Clean up before ending Stata script
*-------------------------------------------------------------------------------

* Keep the index and prediction, then merge onto original data
keep `index' `prediction'
tempfile t1
qui save `t1'
restore
qui merge 1:1 `index' using `t1', nogen
drop `index'

* If y needed encoding, decode
* XX this is inefficient
if "`needs_encoding'"=="yes" {
	tempvar encode1
	encode `yvar', gen(`encode1')
	label values `prediction' `encode1'
	decode `prediction', gen(`prediction'_2)
	drop `prediction'
	rename `prediction'_2 `prediction'
}

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
* Python helper function
*===============================================================================

version 16.0
python:

#-------------------------------------------------------------------------------
# Import required packages and attempt to install w/ Pip if that fails
#-------------------------------------------------------------------------------

# Import required Python modules (pandas, scikit-learn, sfi)
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sfi import Data,Matrix,Scalar
from sklearn import metrics, preprocessing
import numpy as np

# To pass objects to Stata
import __main__

# Set random seed
import random
random.seed(50)

#-------------------------------------------------------------------------------
# Define Python function: run_random_forest
#-------------------------------------------------------------------------------

def run_random_forest(type,vars,n_estimators,criterion,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_features,max_leaf_nodes,min_impurity_decrease,bootstrap,oob_score,n_jobs,random_state,verbose,warm_start,class_weight,prediction,training,importance,nonempty_test, standardize):

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
	
    # Initialize random forest regressor (if model type is regress)
	if type=="regress":
		model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)

	# Initialize random forest classifier (if model type is classify)
	if type=="classify":
		model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)

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
	__main__.model_object  = model
	__main__.model_predict = model_predict
	__main__.features = features
		
	#-------------------------------------------------------------
	# Get model diagnostics (RMSE, classification accuracy, etc)
	#-------------------------------------------------------------
	
	# If regression: get in sample (training sample) mae, rmse
	if type=="regress":
		insample_mae = metrics.mean_absolute_error(y_insample, pred_insample)
		insample_mse = metrics.mean_squared_error(y_insample, pred_insample)
		insample_rmse = np.sqrt(insample_mse)
		Scalar.setValue("e(training_mae)", insample_mae, vtype='visible')
		Scalar.setValue("e(training_rmse)", insample_rmse, vtype='visible')
		Scalar.setValue("e(training_accuracy)",0)

	# If classify: get in sample (training sample) accuracy
	if type=="classify":
		insample_accuracy = metrics.accuracy_score(y_insample, pred_insample)
		Scalar.setValue("e(training_mae)", 0)
		Scalar.setValue("e(training_rmse)", 0)
		Scalar.setValue("e(training_accuracy)", insample_accuracy, vtype='visible')

	# If nonempty test sample, get out of sample stats
	if type=="regress" and nonempty_test==1:
		pred_outsample = model.predict(df_test[features])
		y_outsample = df_test[y]
		outsample_mae = metrics.mean_absolute_error(y_outsample, pred_outsample)
		outsample_mse = metrics.mean_squared_error(y_outsample, pred_outsample)
		outsample_rmse = np.sqrt(outsample_mse)
		Scalar.setValue("e(test_mae)", outsample_mae, vtype='visible')
		Scalar.setValue("e(test_rmse)", outsample_rmse, vtype='visible')
		Scalar.setValue("e(test_accuracy)", 0)

	if type=="classify" and nonempty_test==1:
		pred_outsample = model.predict(df_test[features])
		y_outsample = df_test[y]
		outsample_accuracy = metrics.accuracy_score(y_outsample, pred_outsample)
		Scalar.setValue("e(test_accuracy)", outsample_accuracy, vtype='visible')
		Scalar.setValue("e(test_rmse)", 0)
		Scalar.setValue("e(test_mae)", 0)

	# If applicable, feature importance
	if 1==1:
		feature_importances = DataFrame(model.feature_importances_,
										index = features,
										columns=['importance']).sort_values('importance', ascending=False)
		z = feature_importances.shape
		importance = list(model.feature_importances_)
		Matrix.create("e(importance)", z[0], z[1], -1)
		Matrix.setColNames("e(importance)",['importance'])
		Matrix.setRowNames("e(importance)",list(features.values))
		for i in range(z[0]):
			Matrix.storeAt("e(importance)",i,0,importance[i])
	
end
