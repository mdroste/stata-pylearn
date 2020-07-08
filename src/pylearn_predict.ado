*! Version 0.63, 8jul2020, Michael Droste, mdroste@fas.harvard.edu
*! More info and latest version: github.com/mdroste/stata-pylearn
*===============================================================================
* FILE: pylearn_predict.ado
* PURPOSE: Enables post-estimation -predict- command to obtain fitted values
*   from pylearn
*===============================================================================

program define pylearn_predict, eclass
	version 16.0
	syntax anything(id="argument name" name=arg) [if] [in], [pr xb]
	
	* Mark sample with if/in
	marksample touse, novarlist
	
	* Count number of variables
	local numVars : word count `arg'
	if `numVars'!=1 {
		di as error "Error: More than 1 prediction variable specified"
		exit 1
	}
	
	* Define locals prediction, features
	local predict_var "`arg'"
	local features "${features}"
	
	* Check to see if variable exists
	cap confirm new variable `predict_var'
	if _rc>0 {
		di as error "Error: prediction variable `predict_var' could not be created - probably already exists in dataset."
		di as error "Choose another name for the prediction."
		exit 1
	}
	
	* Generate an index variable for merging
	tempvar temp_index
	gen `temp_index' = _n
	tempfile t1
	qui save `t1'
	
	
	* Also only keep joint nonmissing over features
	foreach v of varlist `features' {
		qui drop if mi(`v')
	}

	* Get predictions
	python: post_prediction("`features'","`predict_var'")
	
	* If post_prediction didnt throw an error
	if import_success==1 {

		* Keep only prediction and index
		keep `predict_var' `temp_index'
		tempfile t2
		qui save `t2'
		
		* Load original dataset, merge prediction on
		qui use `t1', clear
		qui merge 1:1 `temp_index' using `t2', nogen
	}

	* Keep only if touse
	qui replace `predict_var'=. if `touse'==0
	
	
end

python:

# Import SFI, always with stata 16
from sfi import Data,Matrix,Scalar

def post_prediction(vars, prediction):

	# Start with a working flag
	Scalar.setValue("import_success", 1, vtype='visible')

	# Import model from Python namespace
	try:
		from __main__ import model_predict as pred
	except ImportError:
		print("Error: Could not find estimation results. Run a pylearn command before loading this.")
		Scalar.setValue("import_success", 0, vtype='visible')
		return

	# Generate predictions (on both training and test data)
	pred    = pred
	
	# Export predictions back to Stata
   	Data.addVarFloat(prediction)
	Data.store(prediction,None,pred)
	
end