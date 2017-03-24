Intro to Data Visualization in Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958). 

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 Python & Pip](#01-python--pip)
	+ [0.2 Other](#02-other)
	+ [0.3 Virtual Environment](#03-virtual-environment)
- [1.0 Introduction](#10-introduction)
	+ [1.1 Python Modules](#11-python-modules)
- [2.0 Basic Plots](#20-basic-plots)
	+ [2.1 Line Plots](#21-line-plots)
	+ [2.2 Scatter Plots](#22-scatter-plots)
	+ [2.3 Histograms](#23-histograms)
- [3.0 Matplotlib](#30-matplotlib)
	+ [3.2 Pyplot](#32-pyplot)
- [4.0 Seaborn](#40-seaborn)
- [5.0 ggplot](#50-ggplot)


## 0.0 Setup

This guide was written in Python 3.5.

### 0.1 Python & Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).


### 0.2 Other

We'll be needing different modules. Let's install the ones we'll need for this tutorial. Open up your terminal and enter the following commands to install the needed python modules: 

```
pip3 install ggplot
pip3 install seaborn 
pip3 install requests
pip3 install bokeh
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
vim matplotlibrc
```
And then, write `backend: TkAgg` in the file. Now you should be set up with your virtual environment!

Cool, now we're ready to start! 


## 1.0 Introduction

Data Visualization is an important and exciting aspect of data science. It reveals information we otherwise wouldn't have noticed. It allows us to showcase the work we've done through visualizations, which can be stagnant or interactive. 

### 1.1 Python Modules

[matplotlib](http://matplotlib.org/) is a 2D python plotting library which allows you to generate plots, histograms, power spectra, bar charts, errorcharts, scatterplots, etc, with just a few lines of code.

[bokeh](http://bokeh.pydata.org/en/latest/) is an interactive visualization library for modern web browsers presentation. 

[seaborn](http://seaborn.pydata.org/introduction.html#introduction) is a library for making statistical graphics in Python. It's built on top of matplotlib and is tightly integrated with the PyData stack, including support for numpy and pandas data structures and statistical routines from scipy and statsmodels. 

[ggplot](http://ggplot.yhathq.com/) is a plotting system built for making profressional-looking plots quickly with minimal code.


## 2.0 Matplotlib

`matplotlib.pyplot` is a collection of functions that make matplotlib work similar to matlab. Each pyplot function makes some change to a figure. In `matplotlib.pyplot` various states are preserved across function calls, so that it keeps track of things like the current figure and plotting area, and the plotting functions are directed to the current axes. 


### 2.3 Basic Plots

In this section, we'll overview the basic plot types: line plots, scatter plots, and histograms.

#### 2.3.1 Line Plots 

Line graphs are plots where a line is drawn to indicate a relationship between a particular set of x and y values.

``` python
import matplotlib.pyplot as plt
```

To be able to plot anything, we need to provide the data points, so we declare those as follows:

``` python
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
```

Using matplotlib, we can plot a line between plot x and y.
```
plt.plot(x, y)
```

And as always, we use the `show()` method to have the visualizations pop up.
``` python
plt.show()
```

#### 2.3.2 Scatter Plots

Alternatively, you might want to plot quantities with 2 positions as data points. To do this, you first have to import the needed libraries, as always. We'll be using the same data from before:

```  python
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
```

Next, we plot it with the `plt.plot()` method. Note that the `o` denotes the type of graph we're plotting. 

``` python
plt.plot(x, y, 'o')
```

As always, let's look at what we made:

``` python
plt.show()
```

#### 2.3.3 Histograms

Histograms are very often used in science applications and it's highly likely that you will need to plot them at some point. They are very useful to plot distributions. As before, we'll use numpy and matplotlib.

``` python
import numpy as np
import matplotlib.pyplot as plt
```

First, we'll make the data to plot. We're going to make a normal distribution with 1000 points. 

``` python
data = np.random.normal(5.0, 3.0, 1000)
```

Now, we actually make that histogram of the data array and attach a label:

``` python
plt.hist(data)
```

Lastly, let's look at what we've made: 

``` python
plt.show()
```


## 3.0 Customization

The default customization for matplotlib isn't very appealing or even helpful in data visualization tasks. 


### 3.1 Colors

When there are multiple data points or objects, they have to be able to be differentiated between one another. An easy way to do that is by using different marker styles and colors. You can do this by editing the third parameter to include a letter that indicates the color, such as: 

``` python
plt.plot(x, y, "r")
```
This will give you the same line as before, but now it'll be red. 

### 3.2 Styles

You can also customize the style of the your lines and markers. With line graphs, you can change the line to be dotted, dashed, etc, for example the following should give you a dashed line now:

``` python
plt.plot(x,y, "--")
```

You can find other linestyles you can use can be found on the [Matplotlib webpage](http://
matplotlib.sourceforge.net/api/pyplot)

``` python
plt.plot(x,y, "b*")
```

With Scatter Plots, you can customize the dots to be squares, pentagons, etc. This will get you the a scatter plot with star markers:

``` python
plt.plot(x,y, "*")
```

### 3.3 Labels

We want to always label the axes of plots to tell users what they're looking at. You can do this in matplotlib, very easily:

If we want to attach a label on the x-axis, we can do that with the `xlabel()` function: 
``` python
plt.xlabel("X Axis")
```

With a quick modification, we can also do that for the y-axis:
``` python
plt.ylabel("Y Axis")
```

What good is a visualization without a title to let us know what the visualization is showing? Luckily, matplotlib has a built in function for that:
``` python
plt.title("Title Here")
```

Lastly, we can even customize the range for the x and y axes: 
``` python
plt.xlim(0, 10)
plt.ylim(0, 10)
```


## 4.0 Seaborn


``` python
import matplotlib.pyplot as plt
import numpy as np
from urllib.request import urlretrieve
import pandas as pd
```

``` python
url = "https://gist.githubusercontent.com/jhamrick/cfa18fcd3032ba435ec78a194b1447be/raw/4a4052c56161df8e454a61ab5286a769799c64b8/task_data.csv"
urlretrieve(url, "task_data.csv")
```

``` python
task_data = pd.read_csv("task_data.csv")
task_data.head()
```

``` python
fig, ax = plt.subplots()
```    

Plot the bars
``` python
tasks = task_data.groupby(['robot', 'inference'])['robot_tasks'].mean()
ax.bar(np.arange(len(tasks)), tasks, align='center')
```
Show the 50% mark, which would indicate an equal number of tasks being completed by the robot and the human. There are 39 tasks total, so 50% is 19.5

``` python
ax.hlines(19.5, -0.5, 5.5, linestyle='--', linewidth=1)
```    
Set a reasonable y-axis limit
``` python
ax.set_ylim(0, 40)
```    
Apply labels to the bars so you know which is which
``` python
ax.set_xticks(np.arange(len(tasks)))
ax.set_xticklabels(["\n".join(x) for x in tasks.index])
```


``` python
plt.show()
```

And we get: 

![alt text](https://github.com/lesley2958/intro-data-viz/blob/master/figure1.png?raw=true "Logo Title Text 1")

``` python
import seaborn as sns
```

Now, if we re-run the code, we'll see the different default graph for seaborn:

``` python
fig, ax = plt.subplots()
ax.bar(np.arange(len(tasks)), tasks, align='center')
ax.hlines(19.5, -0.5, 5.5, linestyle='--', linewidth=1)

ax.set_ylim(0, 40)
ax.set_xticks(np.arange(len(tasks)))
ax.set_xticklabels(["\n".join(x) for x in tasks.index])
plt.show()
```

And we get: 

![alt text](https://github.com/lesley2958/intro-data-viz/blob/master/figure2.png?raw=true "Logo Title Text 1")

The barplot function gives us a legend telling us which color corresponds to which inference type. But, for this plot, I'd actually like to put the inference types under the bars as ticklabels, and then label each group of bars with the robot type. I can accomplish this by splitting the plot into three subplots, which is quite easy to do using Seaborn's FacetGrid:

``` python
g = sns.FacetGrid(
	task_data,
	col="robot",
	col_order=["fixed", "reactive", "predictive"],
	sharex=False)

    # Create the bar plot on each subplot
g.map(
	sns.barplot,
	"robot", "robot_tasks", "inference",
	hue_order=["oracle", "bayesian"])

    # Now I need to draw the 50% lines on each subplot
    # separately
axes = np.array(g.axes.flat)
for ax in axes:
	ax.hlines(19.5, -0.5, 0.5, linestyle='--', linewidth=1)
	ax.set_ylim(0, 40)

plt.gcf()
```

``` python
plt.show()
```
So we get: 

![alt text](https://github.com/lesley2958/intro-data-viz/blob/master/figure3.png?raw=true "Logo Title Text 1")


## 5.0 ggplot


``` python
from ggplot import *

ggplot(aes(x='date', y='beef'), data=meat) +\
    geom_line() +\
    stat_smooth(colour='blue', span=0.2)
```



