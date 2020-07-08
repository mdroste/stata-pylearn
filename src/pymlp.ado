*! Version 0.63, 8jul2020, Michael Droste, mdroste@fas.harvard.edu
*! More info and latest version: github.com/mdroste/stata-pylearn
*===============================================================================
* Program:   pymlp.ado
* Purpose:   Multi-layer perceptron (neural net) classification/regression in Stata
*            16+ with Python and scikit-learn. Component of pylearn.
*===============================================================================

program define pymlp, eclass
version 16.0
syntax varlist(min=2) [if] [in] [aweight fweight], ///
	[ ///
		type(string asis)            	 /// model type: classifier or regressor
		hidden_layer_sizes(string asis)	 /// Tuple with number of hidden layers and their 
		activation(string asis)			 /// Activation function
		solver(string asis)				 /// Optimization algorithm
		alpha (real 0.0001)			 	 /// L2 penalty parameter (default 0.0001)
		batch_size                       /// Size of minibatches for sotchastic optimizers (default: auto)
		learning_rate(string asis)		 /// learning rate schedule for weight updates
		learning_rate_init(real 0.001) 	 /// initial learning rate (for sgd or adam solvers)
		power_t							 /// learning rate schedule for weight updates
		max_iter(integer 200)			 /// max iterations for neural nets
		shuffle							 /// learning rate schedule for weight updates
		random_state(string asis)		 /// random seed
		tol(real 1e-4) 					 /// tolerance threshold
		verbose 						 /// verbosity of python
		warm_start 						 /// xx not implemented
		momentum(real 0.9)				 /// momentum
		nesterovs_momentum 				 /// nesterovs momentum
		early_stopping 				 	 /// early stopping rule
		validation_fraction(real 0.1)	 /// fraction of training data to set aside for early stopping
		beta_1(real 0.9) 		 		 /// exp parameter in adams smoother, 1st order moments
		beta_2(real 0.999) 				 /// exp parameter in adams smoother, 2nd order moments
		epsilon(real 1e-8) 				 /// some parameter
		n_iter_no_change(real 10) 		 /// max iterations w no change in loss
		standardize                      /// Standardize features
	]

*-------------------------------------------------------------------------------
* Before doing anything: make sure we have Python 3.0+ and good modules
*-------------------------------------------------------------------------------

pylearn, check

*-------------------------------------------------------------------------------
* Handle arguments
*-------------------------------------------------------------------------------

*--------------
* type: string asis, either classify or regress
if "`type'"=="" {
	di as error "ERROR: type() option needs to be specified. Valid options: type(classify) or type(regress)"
	exit 1
}
if ~inlist("`type'","classify","regress") {
	di as error "Syntax error: invalid choice for type (chosen: `type'). Valid options are classify or regress"
	exit 1
}

*--------------
* hidden_layer_sizes: need to validate
if "`hidden_layer_sizes'"=="" local hidden_layer_sizes "10"
local layer_str "`hidden_layer_sizes'"
local layer_str = subinstr("`layer_str'",","," ",.)
local num_layers = wordcount("`layer_str'")
tokenize "`layer_str'"
forval i=1/`num_layers' {
	local nodes_per_layer "`nodes_per_layer', ``i''"
}
local nodes_per_layer = substr("`nodes_per_layer'",3,.)
if `num_layers'>1  local hidden_layer_sizes = "(" + "`hidden_layer_sizes'" + ")"
if `num_layers'==1 local hidden_layer_sizes = "(" + "`hidden_layer_sizes'" + ",)"

*--------------
* activation: activation function choice
if "`activation'"=="" local activation "relu"
if ~inlist("`activation'","identity","logistic","tanh","relu") {
	di as error "Syntax error: activation() must be one of: identity, logistic, tanh, or relu (was `activation')"
	exit 1
}

*--------------
* solver: solver for weight optimization
if "`solver'"=="" {
	local no_solver_specified = 1
	local solver "adam"
}
if ~inlist("`solver'","lbfgs","sgd","adam") {
	di as error "Syntax error: solver() must be one of: lbfgs, sgd, or adam (was `max_depth')"
	exit 1
}

*--------------
* alpha
* xx

*--------------
* batch size: size of minibatches for stochastic optimizers
if "`batch_size'"=="" local batch_size "auto"
if "`batch_size'"!="auto" {
	* xx check to make sure an integer
}

*--------------
* learning_rate: learning rate schedule for weight updates
if "`learning_rate'"=="" local learning_rate "constant"
if ~inlist("`learning_rate'","constant","invscaling","adaptive") {
	di as error "Syntax error: learning_rate() must be one of: constant, invscaling, adaptive (was `learning_rate')"
	exit 1
}

*--------------
* learning_rate_init: controls step size in weight updates, if solver is sgd or adam
if "`learning_rate_init'"=="" local learning_rate_init 0.001
* xx make sure positive number

*--------------
* power_t: exponent for inverse scaling learning rate, if solver is sgd and learning_rate is invscaling
if "`power_t'"=="" local power_t 0.5
* xx make sure positive number? is that required?

*--------------
* max_iter: Max number of iterations
if "`max_iter'"=="" local max_iter 200
if `max_iter'<1 {
	di as error "Syntax error: max_iter() needs to be a positive integer (was `max_iter')"
	exit 1
}

*--------------
* shuffle: Whether to shuffle samples in each iteration, if solver=sgd or adam
if "`shuffle'"=="" local shuffle True

*--------------
* power_t: exponent for inverse scaling learning rate, if solver is sgd and learning_rate is invscaling
if "`power_t'"=="" local power_t 0.5
* xx make sure positive number? is that required?

*--------------
* random_state: initialize random number generator
if "`random_state'"=="" local random_state None
if "`random_state'"!="" & "`random_state'"!="None" {
	if `random_state'<1 {
		di as error "Syntax error: random_state should be a positive integer."
		exit 1
	}
	set seed `random_state'
}

*--------------
* tol: tolerance threshold for optimizing
if `tol'<=0 {
	di as error "Syntax error: tol() can't be negative (was `tol')"
	exit 1
}

*--------------
* verbose: control verbosity of python output (boolean)
if "`verbose'"=="" local verbose 0
if "`verbose'"=="verbose" local verbose 1

*--------------
* warm_start: Unsupported scikit-learn option used to use pre-existing rf object 
if "`warm_start'"=="" local warm_start False

*--------------
* momentum: momentum for gradient descent update, between 0 and 1
if "`momentum'"=="" local momentum 0.9
if `momentum'<=0 | `momentum'>=1 {
	di as error "Syntax error: momentum should be between 0 and 1: 0 < momentum < 1 (was `momentum')"
	exit 1
}


*--------------
* nesterovs momentum: whether to use nesterovs momentum
if "`nesterovs_momentum'"=="" local nesterovs_momentum True

*--------------
* early stopping: Whether to use early stopping to terminate training when validation not improving
if "`early_stopping'"=="" local early_stopping False

*--------------
* validation_fraction: Proportion of training data to set aside for early stopping validation
* xx

*--------------
* beta_1: Exp decay rate used for 1st moment vector estimates in adam, [0,1).
* xx

*--------------
* beta_2: Exp decay rate used for 2nd moment vector estimates in adam, [0,1).
* xx

*--------------
* epsilon: Value for numerical stability in adam
* xx

*--------------
* n_iter_no_change: Max number of epochs to not meet tol improvement, for sgd or adam solvers
* xx

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
* XX this should be an ado
*-------------------------------------------------------------------------------

* Pass varlist into varlists called yvar and xvars
gettoken yvar xvars : varlist
local num_features : word count `xvars'

* generate an index of original data so we can easily merge back on the results
* xx there is probably a better way to do this... feels inefficient
* xx only needs to be done if saving predictions
tempvar index
gen `index' = _n

* preserve original data
preserve

* restrict sample with if and in
marksample touse, strok novarlist
qui drop if `touse'==0

* if classification: check to see if y needs encoding to numeric
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

* Get number of hidden layers, obs per unit

* REVISIT SOLVER SETTING: if no_solver_specified and num_obs_train<10000, use lbfgs
if `no_solver_specified'==1 & `num_obs_train'<100000 {
	local solver lbfgs
}

*-------------------------------------------------------------------------------
* If type(regress), run regression model
*-------------------------------------------------------------------------------

* Store a macro to slightly change results table
if "`type'"=="regress" local type_str "regression"
if "`type'"=="classify" local type_str "classification"

* Pass options to Python to import data, run MLP, return results
python: run_mlp( ///
	"`type'", ///
	"`training_var' `yvar' `xvars'", ///
	"`training_var'", ///
	`hidden_layer_sizes', ///
	"`activation'", ///
	"`solver'", ///
	`alpha', ///
	"`batch_size'", ///
	"`learning_rate'", ///
	`learning_rate_init', ///
	`power_t', ///
	`max_iter', ///
	`shuffle', ///
	`random_state', ///
	`tol', ///
	`verbose', ///
	`warm_start', ///
	`momentum', ///
	`nesterovs_momentum', ///
	`early_stopping', ///
	`validation_fraction', ///
	`beta_1', ///
	`beta_2', ///
	`epsilon', ///
	`n_iter_no_change', `standardize')

*------------------------------------------------------------------------------------
* Format output
*------------------------------------------------------------------------------------

* Collect results from e matrix
local is_rmse: di %10.4f `e(training_rmse)'
local is_mae: di %10.4f `e(training_mae)'
local os_rmse: di %10.4f `e(test_rmse)'
local os_mae: di %10.4f `e(test_mae)'

* Generate formatted strings to display in Stata terminal
local train_obs_f: di %10.0fc `num_obs_train'
local test_obs_f: di %10.0fc `num_obs_test'
local yvar_fmt = "`yvar'"
if length("`yvar'")>13 local yvar_fmt = substr("`yvar'",1,13) + "..."

* Display output
noi di "{hline 80}"
noi di in ye "Multi-layer perceptron `type_str'"
noi di " "
noi di in gr "{ul:Data}"
noi di in gr "Dependent variable  = " in ye "`yvar_fmt'" _continue
noi di in gr _col(41) "Number of training obs   = " in ye `train_obs_f'
noi di in gr "Number of features  = " in ye `num_features' _continue
noi di in gr _col(41) "Number of validation obs = " in ye `test_obs_f'
noi di in gr "Training identifier = " in ye "`training_di'"
no di in gr  "Standardized        =" in ye "`stdize_fmt'"
noi di " "
noi di in gr "{ul:Neural network options}"
di in gr "Hidden layers:           " in ye `num_layers' 
di in gr "Nodes per layer:         " in ye "`hidden_layer_sizes'"
di in gr "Activation function:     " in ye "`activation'"
di in gr "Alpha (L2 penalty term): " in ye "`alpha'"

noi di " "
noi di in gr "{ul:Optimizer settings}"
di in gr "Solver: " in ye "`solver'"
di in gr "Max iterations:             " in ye "`max_iter'"
di in gr "Batch size:                 " in ye "`batch_size'"
if "`solver'"=="sgd" & "`learning_rate'"=="invscaling" di in gr "Power t:                 " in ye "`power_t'"
if inlist("`solver'","adam","sgd")  di in gr "Initial learning rate:      " in ye "`learning_rate_init'"
if inlist("`solver'","adam") di in gr "Exp decay rate, 1st moment: " in ye "`beta_1'"
if inlist("`solver'","adam") di in gr "Exp decay rate, 2nd moment: " in ye "`beta_2'"
if inlist("`solver'","adam") di in gr "Epsilon (for adam solver):  " in ye "`epsilon'"
noi di " "
noi di in gr "{ul:Output}"

* Only display if convergence achieved
if $pymlp_convergence_err==0 {
	if "`type'"=="regress" {
		noi di in gr "Training RMSE       = " in ye `is_rmse'
		*noi di in gr "Training MAE        = " in ye `is_mae'
	}
	if "`type'"=="classify" {
		noi di in gr "Training accuracy   = " in ye `e(training_accuracy)'
	}
	if "`type'"=="regress" & `nonempty_test'==1 {
		noi di in gr "Test RMSE     = " in ye `os_rmse'
		*noi di in gr "Test MAE      = " in ye `os_mae'
	}
	if "`type'"=="classify" & `nonempty_test'==1 {
		noi di in gr "Validation accuracy = " in ye `e(test_accuracy)'
	}
}
else {
		noi di as err "Error: Convergence not achieved after max iterations (`max_iter')"
		noi di as err "Decrease the number of hidden layers / nodes, increase max iterations,"
		noi di as err "or change the optimizer settings."
		noi di in gr ""
}

noi di " "
noi di in gr "Type {help pymlp:help pymlp} to access the documentation."
noi di "{hline 80}"

*-------------------------------------------------------------------------------
* Clean up before ending
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

* Store as locals
ereturn local predict "pylearn_predict"
ereturn local features "`xvars'"
ereturn local type "`type'"
ereturn scalar N = `num_obs_train'
ereturn scalar N_test = `num_obs_test'
ereturn scalar K = `num_features'



end

*===============================================================================
* Python helper functions
*===============================================================================

version 16.0
python:

#-------------------------------------------------------------------------------
# Import required packages and attempt to install w/ Pip if that fails
#-------------------------------------------------------------------------------

# Import Python libraries
from pandas import DataFrame
import numpy as np
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn import metrics
from sfi import Data,Scalar,Macro
import warnings, sys
from sklearn import exceptions, metrics, preprocessing

# To pass objects to Stata
import __main__

#-------------------------------------------------------------------------------
# Define Python function: run_mlp
#-------------------------------------------------------------------------------

def run_mlp(type, vars, training, hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, standardize):


	#-------------------------------------------------------------
	# Prelim
	#-------------------------------------------------------------

	# Reset pymlp_convergence internal global to 0 
	Macro.setGlobal('pymlp_convergence_err', "0")

	#-------------------------------------------------------------
	# Prepare data
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

	#-----------------------------------
	# Run model
	#-----------------------------------

    # Initialize model
	if type=="regress":
		model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change)
	if type=="classify":
		model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change)

	#-------------------------------------------------------------
	# Fit model, get predictions
	#-------------------------------------------------------------

	# Train model on training data
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		try:
			 model.fit(x_insample, y_insample)
		except Warning:
			exc_type, exc_value, exc_traceback = sys.exc_info()	
			Macro.setGlobal('pymlp_convergence_err', "1")
			print(exc_value)

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
