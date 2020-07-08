{smcl}
{* *! version 0.60 8jul2020}{...}
{viewerjumpto "Syntax" "pylearn##syntax"}{...}
{viewerjumpto "Description" "pyforest##description"}{...}
{viewerjumpto "Options" "pyforest##options"}{...}
{viewerjumpto "Examples" "pyforest##examples"}{...}
{viewerjumpto "Author" "pyforest##author"}{...}
{viewerjumpto "Acknowledgements" "pyforest##acknowledgements"}{...}
{title:Title}
 
{p2colset 5 16 21 2}{...}
{p2col :{hi:pylearn} {hline 2}}Supervised learning algorithms in Stata{p_end}
{p2colreset}{...}

{pstd}
Please run {stata pylearn, upgrade} to upgrade to the latest release.

{pstd}
{bf: pylearn} is a set of Stata commands to perform supervised learning in Stata. These commands all exhibit a common Stata-like syntax for model estimation and post-estimation (i.e., they look very similar to regress). Pylearn currently includes five sets of models:

{p 8 17 2}
{manhelp pytree R:pytree} estimates decision trees. {p_end}

{p 8 17 2}
{manhelp pyforest R:pyforest} estimates random forests. {p_end}

{p 8 17 2}
{manhelp pymlp R:pymlp} estimates multi-layer perceptrons (neural-networks). {p_end}

{p 8 17 2}
{manhelp pyada R:pyada} estimates adaptive boosted trees/regressions (AdaBoost). {p_end}

{p 8 17 2}
{manhelp pygradboost R:pygradboost} estimates gradient boosted trees. {p_end}
 
{marker syntax}{title:Syntax}
 
{p 8 15 2}
{cmd:pylearn}, [{cmd:}{it:options}]
                               
 
{synoptset 32 tabbed}{...}
{synopthdr :options}
{synoptline}
{synopt :{opt u:pgrade}} Upgrade to the latest version of pylearn{p_end}
{synopt :{opt s:etup}} Check to see if Python prerequisites are satisfied{p_end}
{synopt :{opt examples}} Print examples{p_end}
{synoptline}
{p 4 6 2}
{p_end}
 
 
{marker description}{...}
{title:Description}
 
{pstd}
{opt pylearn} is a package that provides estimation functionality for a set of supervised learning algorithms directly from Stata. These commands all share a common 'Stata-like' syntax that will feel very familiar Stata users. 

{pstd}
{opt pylearn} wraps around supervised learning implementations provided by the popular scikit-learn library in Python. This library has been extensively debugged by a large group of Python users. The implementations are relatively stable and efficient. All options available for these models in scikit-learn are available in Stata.

{pstd} 
{opt pylearn} relies critically on the Python integration functionality introduced with Stata 16. Therefore, users will need Stata 16, Python 3.6+, and the scikit-learn library installed in order to run.

 
 
{marker options}{...}
{title:Options}

{phang}
{opt upgrade} upgrades pylearn to the latest release from the project GitHub page.

{phang}
{opt setup} checks to see whether the Python prerequisites for this package are satisfied.

{phang}
{opt examples} prints a few examples using pylearn commands.


{marker author}{...}
{title:Author}
 
{pstd}Michael Droste{p_end}
{pstd}mdroste@fas.harvard.edu{p_end}
 
 
 
{marker acknowledgements}{...}
{title:Acknowledgements}

{pstd}This program owes a lot to the wonderful {browse "https://scikit-learn.org/":scikit-learn} library in Python.


