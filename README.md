# Car-Values-Analysis
 Analysis of a used car sale price dataset, applying the CRISP-DM framework to determine what drives pricing. 
This repository also contains a Jupyter notebook, along with associated data and images, which was used to analyze the data.

### Business Understanding
We would like to understand what features of a used car drives prices. From the dataset containing several characterestics of cars, the goal is to identify the most important features that determine car value. We will do this by first building a Machine Learning model and training it on the data. After cross-validating the model, we will extract the features that are most dominant in accurately predicting car prices.

### Data Understanding
The data consists of several attributes of used cars, including year, mileage, vehicle type, manufacturer and condition. There more than 400,000 data points in all.
* We identify incomplete data and decide upon a reasonable strategy to handle such data. The approach used will depend on the type of feature in question. For example, how we fill in missing values for odometer reading will differ from how we treat missing vehicle type.
* We will also get a feel for how closely correlated the features are to each other. This will help us understand if there are redundant features which do not add significantly to data understanding. 
