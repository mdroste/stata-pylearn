
Installing pylearn prerequisites
=================================

Overview
---------------------------------

pylearn is a set of Stata programs for implementing supervised machine learning algorithms in Stata. It relies on Stata 16's built-in Stata functionality and a few Python libraries. In particular, pyforest requires the popular Python packages scikit-learn and pandas (along with all of their prerequisites).

If you are on Windows, the easiest way to satisfy these prerequisites is to use Anaconda. Anaconda is a distribution of Python that includes all of the necessary prerequisites. You can install Anaconda even if you already have one or more Python installations on your computer. Although Anaconda is also available on Mac OSX, there's an odd bug with Stata and NumPy (a Python package used behind the scenes) that breaks the Python integration on those systems.

Mac users should just work with 'vanilla' Python and install the necessary libraries separately.


Windows: Installation with Anaconda
---------------------------------


1. Download the Stata component of pyforest by typing the following Stata command into the Stata window:

```stata
net install pylearn, from(https://raw.githubusercontent.com/mdroste/stata-pylearn/master/) replace
```

2. Download the most recent verison of Anaconda with Python 3 from [Anaconda's website](https://www.anaconda.com/distribution/#download-section).
   Follow the directions on the installer. Install for 'Just me' rather than 'All users'. 
   *If you're on Windows, take note of the install path, as shown in the figure below.*
   *If prompted, choose to make Anaconda your computer's "default" installation of Python*.

<p align="center"><img src="https://raw.githubusercontent.com/mdroste/stata-pylearn/master/docs/images/fig1.png"></p>

3. Close any open Stata windows you might have, and then open a new one. Type "python query" to see if Stata automatically recognizes your Python installation.

<p align="center"><img src="https://raw.githubusercontent.com/mdroste/stata-pylearn/master/docs/images/fig2.png"></p>

4. If your Stata window looks like the screenshot above, with a file path that includes the word Anaconda, then proceed to step 5. Otherwise, you will need to tell Stata where your Anaconda installation is with the "set python_exec" option. If you are on a Mac, refer to the "Common Issue with Mac Installations" below. 

Make sure to write down the path (you can open the Anaconda installer again if you forgot it) Anaconda installed to, and then use python_exec like so:

<p align="center"><img src="https://raw.githubusercontent.com/mdroste/stata-pylearn/master/docs/images/fig2b.png"></p>

5. Run the program "pylearn, setup" to make sure you have all the prerequisite Python libraries. By default, these all come with Anaconda.

Mac: Installation with Python
---------------------------------

There are a few compatibility issues with some versions of Mac OSX and Anaconda, the Python distribution we recommend for Windows (see 'Problems with Anaconda on Mac' below). 

Therefore, if you are on a Mac, we recommend the following steps:

1. Download the Stata component of pyforest by typing the following Stata command into the Stata window:

```stata
net install pyforest, from(https://raw.githubusercontent.com/mdroste/stata-pyforest/master/) replace
```

2. Download the most recent official release of Python from [this link](https://www.python.org/downloads/). Follow the default options on the installer. 

3. Close any open Stata windows you might have, and then open a new one. Type the following into the Stata terminal:

```stata
set python_exec "/Library/Frameworks/Python.framework/Versions/3.8/bin/python3.8", perm
```

Note that this is a different path than what is used in Windows or with Anaconda.

4. Run the command "pylearn, setup" to make sure you have all the prerequisite Python libraries. 


Installation with Existing Python Installation
---------------------------------

If you already have Python installed, it should be straightforward to get running - follow the same guide as above, but starting from step 3.


Note for Mac Installations with Anaconda
---------------------------------

Some (but not all) versions of Mac OSX install Anaconda to a path like "/Users/(username)/opt/anaconda3". Stata will not be able to recognize this path automatically in step 3. Fortunately, there is an easy fix! Simply type the following into the Stata terminal:
```stata
set python_exec "/Users/(username)/opt/anaconda3/bin/python3", perm
```

Once you've typed that once, you don't need to do it again - Stata will remember this path from now on.

If you're on a Mac and not sure where Anaconda installed, simply run the installer again - it will eventually throw an error telling you the path of the installation, which you can use in the "set python_exec" command above.



Problems with Anaconda on Mac
---------------------------------

A small number of Mac users have encountered compatibility issues involving Anaconda and Python on Stata. Sometimes, running a pylearn program leads a large stream of errors relating to importing libraries. This is a [bug](https://www.statalist.org/forums/forum/general-stata-discussion/general/1537891-failure-of-anaconda-miniconda-python-in-stata-16-1-for-macos) in Stata with Anaconda that seems to be difficult to fix. Instead, you should download the most recent version of Python from the official website [click here](https://www.python.org/downloads/) and then follow the instructions 3-5 above, taking care to set your python path in step (4) to the location of this version of Python, and then running "pylearn, setup".