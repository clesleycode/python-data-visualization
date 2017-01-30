Intro to Data Visualization in Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958). 

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python & Pip](#01-python--pip)
	+ [0.2 Other](#02-other)
	+ [0.3 Virtual Environment](#03-virtual-environment)
- [1.0 Background](#10-background)



## 0.0 Setup

This guide was written in Python 3.5.

### 0.1 Python & Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).


### 0.2 Other

We'll soon get into the difference between packages in R and modules in Python. For now, let's install the ones we'll need for this tutorial. Open up your terminal and enter the following commands to install the needed python modules: 

```
pip install ggplot
pip install nltk
pip install seaborn 
```

### 0.3 Virtual Environment

If you'd like to work in a virtual environment, you can set it up as follows: 
```
pip3 install virtualenv
virtualenv your_env
```
And then launch it with: 
```
source your_env/bin/activate
```

To execute the visualizations in matplotlib, do the following:

```
cd ~/.matplotlib
nano matplotlibrc
```
And then, write `backend: TkAgg` in the file. Now you should be set up with your virtual environment!

Cool, now we're ready to start! 




## 1.0 Background

## 2.0 Matplotlib


### 2.2 Pyplot

`matplotlib.pyplot` is a collection of functions that make matplotlib work similar to matlab. Each pyplot function makes some change to a figure. In matplotlib.pyplot various states are preserved across function calls, so that it keeps track of things like the current figure and plotting area, and the plotting functions are directed to the current axes.


``` python
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
```

Here, matplotlib automatically generates the x values for you in the graph. `plot()` takes an arbitrary number of arguments, so you can feed a list like in the previous example or you feed multiple lists, like here:

``` python
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()
```

The `axis()` function in the example above takes a list of `[xmin, xmax, ymin, ymax]` and specifies the viewport of the axes.





## 3.0 
















 
