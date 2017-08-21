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
- [1.0 Introduction](#10-introduction)
    + [1.1 Less is More](#11-less-is-more)
    + [1.2 Colors Matter](#12-colors-matter)
- [2.0 Seaborn](#20-seaborn)
    + [2.1 Factorplot](#21-factorplot)
    + [2.2 FacetGrid](#22-facetgrid)
    + [2.3 Pairplot](#23-pairplot)
    + [2.4 Jointplot & JointGrid](#24-jointplot--jointgrid)
- [3.0 Bokeh](#30-bokeh)
- [4.0 Advanced Graphs](#40-advanced-graphs)
    + [4.1 Slopegraphs](#41-slopegraphs)
    + [4.2 Circle Packing](#circle-packing)
        * [4.2.1 Packcircles](#421-packcircles)
    + [4.3 Sunbursts](#43-sunbursts)
    + [4.4 Word Clouds](#44-word-clouds)

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
pip3 install seaborn 
pip3 install requests
pip3 install bokeh
pip3 install Image
pip3 install wordcloud
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

Using `matplotlib`, we can plot a line between plot x and y.

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

Histograms are very often used in science applications and it's highly likely that you will need to plot them at some point. They are very useful to plot distributions. As before, we'll use `numpy` and `matplotlib`.

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

[seaborn](http://seaborn.pydata.org/introduction.html#introduction) is a library for making statistical graphics in Python. It's built on top of matplotlib and is tightly integrated with the PyData stack, including support for numpy and pandas data structures and statistical routines from scipy and statsmodels. 

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

Plot the bars:

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

Now let's see our work!

``` python
plt.show()
```

And we get: 

![alt text](https://github.com/lesley2958/intro-data-viz/blob/master/figure1.png?raw=true "Logo Title Text 1")

Now, let's import seaborn and see what happens. 

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

Ggplot is yet another module we can use for data visualizations in Python. 

``` python
from ggplot import *

ggplot(aes(x='date', y='beef'), data=meat) +\
    geom_line() +\
    stat_smooth(colour='blue', span=0.2)
```

Advanced Data Visualization in Python
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958), [Byte Academy](byteacademy.co), and [ADI](adicu.com). 

## Table of Contents

- [0.0 Setup](#00-setup)
+ [0.1 Python & Pip](#01-python--pip)
+ [0.2 Other](#02-other)
+ [0.3 Virtual Environment](#03-virtual-environment)
- [1.0 Introduction](#10-introduction)
+ [1.1 Less is More](#11-less-is-more)
+ [1.2 Colors Matter](#12-colors-matter)
- [2.0 Seaborn](#20-seaborn)
+ [2.1 Factorplot](#21-factorplot)
+ [2.2 FacetGrid](#22-facetgrid)
+ [2.3 Pairplot](#23-pairplot)
+ [2.4 Jointplot & JointGrid](#24-jointplot--jointgrid)
- [3.0 Bokeh](#30-bokeh)
- [4.0 Advanced Graphs](#40-advanced-graphs)
+ [4.1 Slopegraphs](#41-slopegraphs)
+ [4.2 Circle Packing](#circle-packing)
* [4.2.1 Packcircles](#421-packcircles)
+ [4.3 Sunbursts](#43-sunbursts)
+ [4.4 Word Clouds](#44-word-clouds)
- [5.0 Final Words](#50-final-words)


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
pip3 install Image
pip3 install wordcloud
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

Unique data visualizations are more memorable. Most of the visualizations we've seen so far are kind of boring, especially if you have multiple of them. You're very quickly going to lose the attention of your audience. In this tutorial, we'll get into more specific and unique approaches. 

It's important to note that while these approaches definitely have applications, they're not as commonly used as bar graphs, line graphs, scatter plots, or even histograms. These basic visualization types can still be the best choice for certain, straightforward data stories. However, communicating complex topics — hierarchies, longitudinal data, and multi-variable comparisons, and so on — often involves more advanced visualizations. That's our focus for this tutorial. 

But before we go into code, let's overview some strategies for effective data visualization. 

### 1.1 Less is More

Very often, when it comes to plotting, less is more. Data usually looks better "naked". What does this mean?

- Remove backgrounds

- Remove redundant labels

- Lighten your lines, or remove them!

- Reduce unnecessary colors

- Remove special effects


### 1.2 Colors Matter

The default library, matplotlib, is fairly ugly, as we've reviewed before. This is why using modules like seaborn, bokeh, etc, are so popular - their default color schemes are much more appealing. 

Also, keep in mind that many people are color blind, so choosing your color schemes is incredibly important so that your work is accessible to everyone. 

## 2.0 Seaborn

Recall, seaborn is a Python data visualization library with an emphasis on statistical plots. The library is an excellent resource for common regression and distribution plots, but where Seaborn really shines is in its ability to visualize many different features at once.

To showcase Seaborn, we'll use the UCI "Auto MPG" data set. First, we'll need to do a bit of pre-processing using pandas. We import the needed modules, as follows: 

``` python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
sns.set_style('darkgrid')
```

In names, we store the names of the columns we'll be working with:

``` python
names = [
'mpg'
,  'cylinders'
,  'displacement'
,  'horsepower'
,  'weight'
,  'acceleration'
,  'model_year'
,  'origin'
,  'car_name'
]
```

Next, we read in the information from the CSV by feeding it in the link of the dataset and the column names:

``` python
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", sep='\s+', names=names)
```

We make a 'maker' column by taking the car_name column and taking the first word:

``` python
df['maker'] = df.car_name.map(lambda x: x.split()[0])
```

Next, we transform the numerical values in the origin column to categorical values:

``` python
df.origin = df.origin.map({1: 'America', 2: 'Europe', 3: 'Asia'})
```

Then, we drop the rows with missing values. Notice that now we have 392 rows instead of 398.

``` python
df = df.applymap(lambda x: np.nan if x == '?' else x).dropna()
```

And lastly, we convert the numbers in horsepower to floats and show what we've got:

``` python
df['horsepower'] = df.horsepower.astype(float)
df.head()
```

Now, we can begin our visualizations!


### 2.1 Factorplot 

In `seaborn`, you can easily build conditional plots, which allows us to see what the data looks like when segmented by one or more variables We can do this with factorplots. If we wanted to see how cars' MPG has varied over time, we could see this in aggregate with:

``` python
sns.factorplot(data=df, x="model_year", y="mpg")  
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/factorplot1.png?raw=true "Logo Title Text 1")

But we can also segment by, say, region of origin:

``` python
sns.factorplot(data=df, x="model_year", y="mpg", col="origin")
plt.show()  
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/factorplot2.png?raw=true "Logo Title Text 1")


### 2.2 FacetGrid

What's so great factorplot is that rather than having to segment the data ourselves and make the conditional plots individually, Seaborn provides a convenient API for doing it all at once.

The FacetGrid object is a slightly more complex, but also more powerful, take on the same idea. Let's say that we wanted to see KDE plots of the MPG distributions, separated by country of origin:

``` python
g = sns.FacetGrid(df, col="origin")
g.map(sns.distplot, "mpg")
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/facetgrid.png?raw=true "Logo Title Text 1")

Or let's say that we wanted to see scatter plots of MPG against horsepower with the same origin segmentation:

``` python
g = sns.FacetGrid(df, col="origin")  
g.map(plt.scatter, "horsepower", "mpg")  
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/facetgrid2.png?raw=true "Logo Title Text 1")

Using FacetGrid, we can map any plotting function onto each segment of our data. For example, above we gave plt.scatter to g.map, which tells Seaborn to apply the matplotlib plt.scatter function to each of segments in our data. We don't need to use plt.scatter, though; we can use any function that understands the input data. For example, we could draw regression plots instead:

``` python
g = sns.FacetGrid(df, col="origin")  
g.map(sns.regplot, "horsepower", "mpg")  
plt.xlim(0, 250)  
plt.ylim(0, 60)  
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/facetgrid3.png?raw=true "Logo Title Text 1")

We can even segment by multiple variables at once, spreading some along the rows and some along the columns. This is very useful for producing comparing conditional distributions across interacting segmentations:

``` python
df['tons'] = (df.weight/2000).astype(int)  
g = sns.FacetGrid(df, col="origin", row="tons")  
g.map(sns.kdeplot, "horsepower", "mpg")  
plt.xlim(0, 250)  
plt.ylim(0, 60)
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/facetgrid4.png?raw=true "Logo Title Text 1")

### 2.3 Pairplot

While factorplot and FacetGrid are for drawing conditional plots of segmented data, pairplot and PairGrid are for showing the interactions between variables. For our car data set, we know that MPG, horsepower, and weight are probably going to be related; we also know that both these variable values and their relationships with one another, might vary by country of origin. Let's visualize all of that at once:

``` python
g = sns.pairplot(df[["mpg", "horsepower", "weight", "origin"]], hue="origin", diag_kind="hist")  
for ax in g.axes.flat:  
plt.setp(ax.get_xticklabels(), rotation=45)
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/pairplot1.png?raw=true "Logo Title Text 1")

As FacetGrid was a fuller version of factorplot, so PairGrid gives a bit more freedom on the same idea as pairplot by letting you control the individual plot types separately. Let's say, for example, that we're building regression plots, and we'd like to see both the original data and the residuals at once. PairGrid makes it easy:

``` python
g = sns.PairGrid(df[["mpg", "horsepower", "weight", "origin"]], hue="origin")  
g.map_upper(sns.regplot)  
g.map_lower(sns.residplot)  
g.map_diag(plt.hist)  
for ax in g.axes.flat:  
plt.setp(ax.get_xticklabels(), rotation=45)
g.add_legend()  
g.set(alpha=0.5)
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/pairgrid.png?raw=true "Logo Title Text 1")

We were able to control three regions (the diagonal, the lower-left triangle, and the upper-right triangle) separately. Again, you can pipe in any plotting function that understands the data it's given.


### 2.4 Jointplot & JointGrid

The final Seaborn objects we'll talk about are jointplot and JointGrid; these features let you easily view both a joint distribution and its marginals at once. Let's say, for example, that aside from being interested in how MPG and horsepower are distributed individually, we're also interested in their joint distribution:

``` python
sns.jointplot("mpg", "horsepower", data=df, kind='kde')  
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/jointplot.png?raw=true "Logo Title Text 1")

As before, JointGrid gives you a bit more control by letting you map the marginal and joint data separately. For example:

``` python
g = sns.JointGrid(x="horsepower", y="mpg", data=df)  
g.plot_joint(sns.regplot, order=2)  
g.plot_marginals(sns.distplot)  
plt.show()
```

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/jointgrid.png?raw=true "Logo Title Text 1")

## 3.0 Bokeh

Bokeh is a python interactive visualization library that targets modern web browsers for presentation. We'll begin with a simple visualization using the built-in iris data:

``` python
from bokeh.sampledata.iris import flowers
from bokeh.plotting import figure, show, output_file
```

This configures the default output state to generate output saved to a file when `show()` is called.

``` python
output_file("iris.html", title ="iris.py example")
```

This attaches color labels to the species:

``` python
colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
colors = [colormap[x] for x in flowers['species']]
```

This attaches a title to the visualization we're working on. 

``` python
p = figure(title = "Iris Morphology")
```

Finally, we're adding the data! 

``` python
p.circle(flowers["petal_length"], flowers["petal_width"],
color=colors, fill_alpha=0.2, size=10)
```

Finally, let's take a look:

``` python
show(p)
```

This is what we get! An awesome visualization:

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/bokeh_plot.png?raw=true "Logo Title Text 1")


## 4.0 Advanced Graphs

Let's take it up a notch though. 

### 4.1 Slopegraphs

Slopegraphs are a special type of line chart where two or more sets of values are compared by connecting each group's values on one scales to their values on the second scale. To make this more clear, let's look at an example:

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/slopegraph.png?raw=true "Logo Title Text 1")

For reference, you typically use these to compare a groups' rate of change or rank-order between categories.

### 4.2 Circle Packing

Circle packing diagrams show groups as tightly-organized circles, and are often used to show hierarchies where smaller groups are either colored similarly to others in the same category, or nested within larger groups.

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/circle%20packing.png?raw=true "Logo Title Text 1")

#### 4.2.1 Packcircles

The `packcircles` package provides an experimental version of the algorithm via the function circleGraphLayout. To use it, first encode the graph of tangencies as a list of vectors:

``` R
library(packcircles)
library(ggplot2)

internal <- list(
c(9, 4, 5, 6, 10, 11),
c(5, 4, 8, 6, 9),
c(8, 3, 2, 7, 6, 5, 4),
c(7, 8, 2, 1, 6)
)
```

Next, we specify the sizes of all external circles:
``` R
external <- data.frame(id = c(1, 2, 3, 4, 6, 10, 11), radius=10.0)
```

Now, pass these two objects to the circleGraphLayout function which will attempt to find a corresponding arrangement of circles which we can then be plotted:
``` R
layout <- circleGraphLayout(internal, external)
```

Here, we generate circle vertices from the layout for plotting:
``` R
plotdat <- circlePlotData(layout, xyr.cols = 2:4, id.col = 1)
```

And lastly, we draw the circles annotated with their IDs.
``` R
ggplot() +
geom_polygon(data=plotdat, aes(x, y, group=id), fill=NA, colour="brown") +
geom_text(data=layout, aes(x, y, label=id)) +
coord_equal() +
theme_bw()
```

This gets us: 

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/r%20circlegraphs.png?raw=true "Logo Title Text 1")

Not the best looking thing ever, but it gets the general idea.

### 4.3 Sunbursts

Sunbursts show a hierarchical structure in a circular layout, with each ring outward representing a deeper level of the hierarchy. 

While sunbursts share some of the disadvantages of pie charts and are not well-suited to precise size comparisons, they do allow notable segments of a complex, multi-layered hierarchy to be quickly identified to guide further action.

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/sunburst.jpg?raw=true "Logo Title Text 1")

### 4.4 Word Clouds

A word cloud is a visual representation of text data, typically used to depict keyword metadata (tags) on websites, or to visualize free form text. 

We begin by importing the needed python modules:

``` python
import os
from wordcloud import WordCloud
```

Recall that we can use the `os` module to read the text to generate the word cloud:
``` python
d = os.getcwd()
text = open(os.path.join(d, 'constitution.txt')).read()
```

Now, we actually generate a word cloud with the `generate()` method:
``` python
wordcloud = WordCloud().generate(text)
```

Using matplotlib, we'll now generate the actual image:
``` python
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

You can customize the font size:
``` python
wordcloud = WordCloud(max_font_size=40).generate(text)
```

And finally, let's see what we've made:
``` python
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

This is what it looks like:

![alt text](https://github.com/ByteAcademyCo/advanced-dv/blob/master/wordclouds.png?raw=true "Logo Title Text 1")

## 5.0 Final Words

And that wraps up the Advanced Data Visualization portion of this course! As you can see, there are plenty of methods you can choose to visualize your data. This process definitely requires technical skills, but also a bit of creativity and intuition. 

Next up, we'll go into Exploratory Analysis with Data Visualization. 

### 5.1 Resources

Obviously this tutorial doesn't encompass everything there is to learn about Data Visualization. Here are some resources I recommend for learning more:

[DataQuest](https://www.dataquest.io/dashboard) <br>
[Python DV Cookbook](https://www.dropbox.com/s/iybhvjblkjymnw7/Python%20Data%20Visualization%20Cookbook%20-%20Milovanovic%2C%20Igor-signed.pdf?dl=0)


