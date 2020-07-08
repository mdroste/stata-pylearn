*===============================================================================
* FILE: pytree_d.ado
* PURPOSE: Enables post-estimation display of decision tree
* SEE ALSO: pyforest.ado
* AUTHOR: Michael Droste
*===============================================================================

program define pytree_d, eclass
	version 16.0
	
	python: display_tree()
	
	
end

python:

def display_tree():

	# Import tree from main namespace
	from __main__ import tree as tree
	from sklearn.tree import export_text
	
	# Import feature names from main namespace
	from __main__ import features as features
	
	# Grab text representation of tree
	r = export_text(tree, feature_names=(list(features)))

	# XX exception handling for use of predict without previous use of pytree
	
	# Display text representation of tree
	print(r)
	
end