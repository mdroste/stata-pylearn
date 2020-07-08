*! Version 0.62, 8jul2020, Michael Droste, mdroste@fas.harvard.edu
*! More info and latest version: github.com/mdroste/stata-pylearn
*===============================================================================
* FILE:     pylearn.ado
* PURPOSE:  Wrapper/setup program for pylearn, a suite of supervised learning
*           algorithms in Stata.
*===============================================================================

program pylearn
version 16.0
syntax, [Upgrade examples Setup Check]


*-------------------------------------------------------------------------------
* If no options specified, spit out informative message
*-------------------------------------------------------------------------------

if "`upgrade'"=="" & "`examples'"=="" & "`setup'"=="" & "`check'"=="" {
    di "No options were specified. The pylearn program doesn't do anything on its own."
    di "See the help file ({stata help pylearn}) for more information."
}

*-------------------------------------------------------------------------------
* If upgrade specified, try to upgrade from GitHub
*-------------------------------------------------------------------------------

if "`upgrade'"!="" {
   cap net uninstall pylearn
    net install pylearn, from(https://raw.githubusercontent.com/mdroste/stata-pylearn/master/src) replace
}

*-------------------------------------------------------------------------------
* If setup specified, run setup program
*-------------------------------------------------------------------------------

if "`setup'"!="" {

    *-----------------------------------------------------
    * Check for Stata
    *-----------------------------------------------------

    * First: Check to see if we have Python, and it is version 3 or above
    qui python query
    local python_path = r(execpath)
    local python_vers = r(version)
    if "`python_path'"=="" {
        di as error "Error: No python path found! Do you have Python 3.6+ installed?"
        di as error "Please refer to the installation instructions on Github:"
        di as error "https://github.com/mdroste/stata-pylearn/blob/master/docs/install.md"
        exit 1
    }

    *-----------------------------------------------------
    * Check for modules
    *-----------------------------------------------------

    * Start by assuming we have them all
    local has_pandas = 1
    local has_sklearn = 1
    local has_numpy = 1
    local has_sfi = 1

    * Check for pandas
    local modname pandas
    di in gr " "
    di in gr "Looking for Python module `modname'..."
    local installed 0
    cap python which `modname'
    if _rc==0 {
        di in gr "  The module `modname' was found."
    }
    if _rc!=0 {
        di in gr "  Warning: Could not find the module `modname'. "
        local has_pandas = 0
    }

    * Check for numpy
    local modname numpy
    di in gr " "
    di in gr "Looking for Python module `modname'..."
    local installed 0
    cap python which `modname'
    if _rc==0 {
        di in gr "  The module `modname' was found."
    }
    if _rc!=0 {
        di in gr "  Warning: Could not find the module `modname'. "
        local has_numpy = 0
    }

    * Check for sklearn
    local modname sklearn
    di in gr " "
    di in gr "Looking for Python module scikit-learn..."
    local installed 0
    cap python which sklearn
    if _rc==0 {
        di in gr "  The module scikit-learn was found."
    }
    if _rc!=0 {
        di in gr "  Warning: Could not find the module scikit-learn. "
        local has_sklearn = 0
    }

    *-----------------------------------------------------
    * If we need to install anything...
    *-----------------------------------------------------

    * If pandas, numpy, or sklearn not found
    if `has_pandas'==0 | `has_numpy'==0 | `has_sklearn'==0 {
        
        * Look for pip, install if not found
        di in gr "We will try to install the modules we couldn't find using pip."
        di in gr " "
        di in gr "Looking for pip..."
        cap python which `pip'
        if _rc == 0 {
            di in gr "  Pip was found."
            local has_pip=1
        }
        if _rc != 0 {
            di in gr "  Warning: Could not find the module pip."
            di in gr "  Trying to install now."
            cd "`c(sysdir_plus)'"
            copy "https://bootstrap.pypa.io/get-pip.py" get-pip.py, replace
            shell `python_path' get-pip.py
            di in gr "  Installed pip. "
        }

        * Try to install pandas if necessary
        if `has_pandas'==0 {
            local modname pandas
            di in gr "    Trying to install `modname' automatically with pip..."
            sleep 300
            python: install_mod("`python_path'","`modname'")
            if `pf_install'==0 {
                di as error "  Error: Could not install `modname' automatically. You may need to install it manually."
                di as error "  Please see the help file ({help pylearn}) for more info."
                exit 1
            }
        }

        * Try to install numpy if necessary
        if `has_numpy'==0 {
            local modname numpy
            di in gr "    Trying to install `modname' automatically with pip..."
            sleep 300
            python: install_mod("`python_path'","`modname'")
            if `pf_install'==0 {
                di as error "  Error: Could not install `modname' automatically. You may need to install it manually."
                di as error "  Please see the help file ({help pylearn}) for more info."
                exit 1
            }
        }

        * Try to install sklearn if necessary
        if `has_sklearn'==0 {
            local modname scikit-learn
            di in gr "    Trying to install `modname' automatically with pip..."
            sleep 300
            python: install_mod("`python_path'","`modname'")
            if `pf_install'==0 {
                di as error "  Error: Could not install `modname' automatically. You may need to install it manually."
                di as error "  Please see the help file ({help pylearn}) for more info."
                exit 1
            }
        }

    }

    *-----------------------------------------------------
    * Wrap up
    *-----------------------------------------------------
        
    di in gr " "
    di in gr "Done! All prerequisites for pylearn are installed."
    di in gr "You may need to restart Stata for any installed Python modules to become available."
    di in gr " "



}

*-------------------------------------------------------------------------------
* If check specified, check for things
*-------------------------------------------------------------------------------

if "`check'"!="" {
    
    * Define a local for missing prereqs
    local missing_prereqs = 0

    * First: Check to see if we have Python, and it is version 3 or above
    qui python query
    local python_path = r(execpath)
    local python_vers = r(version)
    if "`python_path'"=="" {
        di as error "Error: No python path found! Do you have Python 3.6+ installed?"
        di as error "Please refer to the installation instructions on Github:"
        di as error "https://github.com/mdroste/stata-pylearn/blob/master/docs/install.md"
        exit 1
    }

    * Check to see if we have NumPy
    cap python which numpy
    if _rc!=0 {
        di as error "Error: Could not import the Python module numpy. "
        local missing_prereqs = 1
    }

    * Check to see if we have Pandas
    cap python which pandas
    if _rc!=0 {
        di as error: "Error: Could not import the Python module pandas."
        local missing_prereqs = 1
    }

    * Check to see if we have Scikit-learn
    cap python which sklearn
    if _rc!=0 {
        di as error: "Error: Could not import the Python module scikit-learn (sklearn)."
        local missing_prereqs = 1
    }

    * Check to see if we have SFI (definitely should have this, comes w/ Stata 16)
    cap python which sfi
    if _rc!=0 {
        di as error: "Error: Could not import the Python module sfi."
        di as error: "This is weird, since it should come with Stata 16..."
        local missing_prereqs = 1
    }

    * If missing prereqs, post an error
    if `missing_prereqs'==1 {
        di as error: "One or more packages was not found (see above)."
        di as error: "Use the command {pylearn, setup: pylearn, setup} to try to install these automatically."
        di as error: "Alternative, use a Python package management tool like pip to install them yourself."
        exit 1
    }

}

*-------------------------------------------------------------------------------
* If example specified, run examples
*-------------------------------------------------------------------------------

if "`examples'"!="" {
    di "See GitHub for examples"
}

end

*===============================================================================
* Helper function for python subprocess check call
*===============================================================================

python:

import subprocess
from sfi import Macro

def install_mod(python_path, package):
    try:
        subprocess.check_call([python_path, "-m", "pip", "install", "--user", package])
        Macro.setLocal('pf_install', '1')
    except subprocess.CalledProcessError:
        Macro.setLocal('pf_install', '0')
    
end