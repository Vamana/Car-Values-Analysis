# Car Values Analysis
 We present an analysis of a used car sale price dataset, applying the CRISP-DM framework to determine what drives pricing. 
This repository also contains a Jupyter notebook,  along with associated data and images, which was used to analyze the data.

### Business Understanding
We would like to understand what features of a used car drives prices. From the dataset containing several characterestics of cars, the goal is to identify the most important features that determine car value. We will do this by first building a Machine Learning model and training it on the data. After cross-validating the model, we will extract the features that are most dominant in accurately predicting car prices.

### Data Understanding
The data consists of several attributes of used cars, including year, mileage, vehicle type, manufacturer and condition. There more than 400,000 data points in all.
* We identify incomplete data and decide upon a reasonable strategy to handle such data. The approach used will depend on the type of feature in question. For example, how we fill in missing values for odometer reading will differ from how we treat missing vehicle type.
* We will also get a feel for how closely correlated the features are to each other. This will help us understand if there are redundant features which do not add significantly to data understanding. 

### Data Preparation
Before any analysis and model building can be done, we first clean up the data, remove outliers and fill in missing data. We will also drop some features from our analysis as being secondary to influencing pricing.

### Data Cleanup Stage 1:
We drop features that are not useful characterestics for modeling. We also drop regional data since we want to do a nation-wide analysis and pinpoint those features that are applicable to all regions.
The features we drop at this stage are id', 'VIN', 'region', and 'state'.
#### Outliers
We remove outliers (data with unusually high or unsusually values). We removed data with very high or very low mileage, and very high or very low sale price.

### Data Cleanup Stage 2: Missing Data.
Fill in missing data. The strategy used to fill in missing values will depend on the feature being addressed.
#### Proportional Imputer
For columns 'condition', 'cylinders', 'drive', and 'size', at least <b>30% of the values are missing. </b> it will be misleading to impute a default value or even a mean value for missing values; this will skew the data during cleanup! <br> 
We will fill in missing values using a novel <b> Proportional Imputer function </b>: the value to be filled will be drawn from the existing values *with a probability equal to the occurrence of the values*.<br> 
For example, if we have values \['A', 'B' 'C'\] occurring with probabilities \[0.25, 0.3, 0.45\] then the filled value will be 'A' 25% of the time, 'B' 30% of the time and 'C' 45% of the time.

For other features that only have a few missing values, we use a SimpleImputer that will fill in missing values with the median value or an attribute like 'other'.

### Data Cleanup Stage 3:
We Convert 'cylinder' information to numeric values. We also drop 'manufacturer', 'title_status' and 'fuel'

#### Correlation between numeric features

![CorrHeatmap](https://github.com/Vamana/Car-Values-Analysis/assets/7783577/48fcc1da-f169-4fb8-841a-4f0e9376bf44) <br/>
The numeric features 'year', 'odometer' and 'cylinders' are not highly correlated with each other. There is a rather weak negative correlatiion between year and odometer.

### Modeling
We now build a Machine Learning (ML) model to learn from the data and predict the price of a used car.
We split the data into two sets: a training dataset and a test dataset which we will use to validate our predictive model.

We notice that the target variable (price), has a long tail even after removing some outliers, so we will use logarithm of the sale price to get an approximately normal distribution, which is better suited numerically for model building. This is done by using a TransformedTargetRegressor in the modeling.
#### Ridge Model with Grid Search
After encoding categorical features with a OneHotEncoder, we build a Ridge estimator model with grid search to find the best alpha (Ridge model hyperparameter).
After training the data on the model, we find that the best estimator performs as well on the training data as on the test data (the mean squared error is very similar on both datasets).
The predicted values vs test data values (for price) are shown below:
![TestVsPredictedPrice](https://github.com/Vamana/Car-Values-Analysis/assets/7783577/feeedd58-b0e5-4205-baf4-422501146b18) <br/>
We see that the scatter plot is symmetric about the 45 degree line, as expected.

We are now in a position to extract the most important features that positively or negatively affect the pricing of a used vehicle. This is done by considering the features with the largest (absolute value) Ridge coefficients. Recall that a large (positive or negative) coefficient indicates that the feature has a large influence on sales price. <b>We consider a feature to be 'important' if its Ridge coefficient is at least 30% of the largest coefficient.</b> We will discuss the results of the feature extraction in the Recommendations section below.

### Evaluation
We build another model which uses a different strategy to extract feature importance, called a LASSO selector, in order to validate the results of the Ridge model above.
#### LASSO selector
We build a LASSO selector to select the 10 most important features (positive or negative) to verify that the ridge model fidings above are reasonable. We run the LASSO selector along with the Ridge model (with grid search for the alpha hyperparameter) as earlier. We find that the results from LASSO are very similar to the Ridge model, validating our earlier results. Details are in the linked Jupyter notebook.

### Recommendations
Based on the results of our Machine Learning model, we can now make some recommendations to car dealers. Some of these recommendations are probably expected from past experience, but we are also able provide some possibly new insights that we have uncovered.

#### The most important desirable features
These are the features that contribute most to increased sale value of a vehicle. <br/>
- <b> Condition </b> (New, excellent, good). Buyers are willing to pay a premium for a vehicle in good to excellent condition.
- <b> Model Year.</b> The later the model year, the more buyers are willing to pay for the vehicle.
- <b> Certain Vehicle Types </b> (pickup trucks!) Customers value pickup trucks more than any other vehicle type.
- Interestingly, <b>number of cylinders</b> (which influences how powerful the vehicle is) is <b>*not*</b> considered to be very important at all from our analysis. <br/>
Below is a bar plot  of the influence of the most desirable features on car price.
![DesirableFeatures](https://github.com/Vamana/Car-Values-Analysis/assets/7783577/388438b8-a650-4612-a999-c94bf00575ed) <br/>

#### The most important *undesirable* features
These are the features that contribute most to the <b>*decrease*</b> in sale value of a vehicle. <br/>

- <b> Odometer (mileage) </b> The higher the mileage of the vehicle, the less buyers are willing to pay. We note that there is only a mild negative correlation between model year and odometer, so the undesirability of high mileage goes beyond correlation with model year.
- <b> Condition </b> (salvage, fair). We note that even fair condition is not acceptable to buyers, perhaps because of 'condition padding' by sellers.
- <b> Certain Vehicle Types </b> (hatchbacks, minivans, sedans, station wagons). It is interesting to note that buys are not willing to pay a premium for sedans.
- <b> SUVs </b> While SUVs are the highest-selling *new* cars, we note that they are neither strongly desirable or undesirable in the *used* car market. <br/>
Below is a bar plot  of the influence of the most undesirable features on car price.
![UndesirableFeatures](https://github.com/Vamana/Car-Values-Analysis/assets/7783577/20ba7167-d0b1-48de-9783-372e49ec9196) <br/>

### Next Steps
Now that we have a good understanding of the important features that drive used car price, we consider some further steps to sharpen our analysis. <br/>
- It will be useful to see if the inclusion of some of the dropped features, notably manufacturer, has an effect on sales price.
- A separate analysis for each state or region might reveal regional differences in customer preference.
- It will be interesting to include some nonlinearity, especially in year and odometer, in our regression model to see if we get a tighter predictive model.







