# LASSO-Regression-Implementation

***Intro***

This Python script implements LASSO Regression using a weather dataset taken from Ted Stevens Anchorage International Airport in Anchorage, Alaska, dated from 1/1/1973 to 1/1/2023. This dataset was obtained by querying the NOAA's Climate Data Online (CDI) tool at this address https://www.ncdc.noaa.gov/cdo-web/. 

***Versions***

This code was developed with VSCode version 1.84.2 and Python version 3.11.2.

***Dependencies***

The following Python modules need to be installed:

* Pandas
* NumPy
* Matplotlib
* Scikit-learn

***Features***
* Implementation of LASSO Regression algorithm
* Testing of various hyperparameters (alpha, lambda, iterations of gradient descent)
* Plotting Mean Squared Error (MSE) against the values of hyperparameters listed above

***Instructions***

* Download the lasso.py script contained within this repository.
* Download the Anchorage.csv file contained within this repository. Anchorage.csv is simply referenced as Anchorage.csv within the script because it was located within the same directory when we developed this code. Make sure that you correctly reference the path to Anchorage.csv before executing the script, otherwise the file will not be found. 
* Simply run the script, and various tests will be executed with plots to accompany them after they are complete (the values of the hyperparameter vs the MSE). First, alpha values are tested to find the optimal learning rate. Next, lambda values are tested to determine the optimal regularization coefficient.
Finally, various numbers of iterations of gradient descent are tested to find the optimal number of iterations.
* Do note that when we ran the script, when a plot pops up after a test, the script pauses until you exit out of the plot window, so make sure you exit the pop-up after looking at/saving the plot. This script takes a while to run, but there will be updates
on its status within the console, so you can be assured it is running.


