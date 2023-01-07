#  Seasonal Flu Vaccination
*Prepared By Kit N., Lucas F. and Navpreet N.* <br> *12 January 2022*

## Introduction

This exercise aims to predict whether individuals will receive their seasonal flu vaccine. To create the predictive model, we fit a machine learning model with random forest classifier (the baseline) to the vaccination data collected in the United States National 2009 H1N1 Flu Survey (NFHS)[^1]. 

Once the baseline is established, we explore various tuning techniques includes:
* feature selection to reduce the number of input features,
* hyperparameters tuning with RandomisedSearchCV, and
* hyperparametrs tuning with GridSearchCV.

A second machine learning model involving deep learning is considered. 

We then compare the performance of each model to determine which has sufficient predictive power. *To add - discuss performance metric and what will indicate stronger performance*

[^1]: The NFHS was one-time survey designed to monitor vaccination during 2009-2010 flu season in response to the 2009 H1N1 pandemic. For further information, refer to [U.S. National 2009 H1N1 Flu Survey (NFHS)](https://webarchive.loc.gov/all/20140511031000/http://www.cdc.gov/nchs/nis/about_nis.htm#h1n1)


## The Dataset
 The NFHS data contains 26,707 survey respondents collected between late 2009 amd early 2010. Each survey respondents were asked whether they had received H1N1 and seasonal flu vaccines and additional questions about themselves such as social, economic and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviours towards mitigating tranmission. 

 This dataset is a copy shared by the United States National Center for Health Statistics for a data science practice competition hosted by [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/).

 Given the focus of this exercise is on seasonal flu vaccination prediction, H1N1 specific data included in the NFHS data is dropped and not used for fitting machine learning model. After removing H1N1 specific variables, the starting point is a dataset with 30 columns. The first column "respondent_id" is a unique identifier. The remaining columns include answer to additional questions asked in the survey. [Appendix 1](#appendix-1) describes the features in detail. <br>
 
### Data checking and cleaning
The following summarises the data checking and cleaning performed:
* **Class balance** <br>
<img src="./Diagram/chk_balance_class.jpg" alt="drawing" width="280" height = "100"/> <br>
"seasonal_vaccine" target variable has balanced class.

* Missing value


### Exploring the data


## Build and Train Machine Learning Model
### Establish the baseline model
We decided to use the random forest machine learning model as our baseline for several reasons. First, random forests are known for their high accuracy and ability to handle large and complex datasets. This was important for us as we were working with a dataset that has a high number of features.

Second, random forests are robust to overfitting. This means the models are less likely to produce poor generalisation performance when applied to new data. This was an important consideration for our project as we wanted to ensure the model would be able to generalise well to new examples and not just perform well on the training data. 

Last, random forest models are relatively simple to implement and do not require much fine-tuning, which we decided made it a good choice for a baseline model.
### Tune the baseline
### i. Features Selection
The first method of tuning our model was to limit the number of features we trained the model on. To select which features to include, we computed a correlation matrix to consider which features had the highest positive correlation to the target variable (whether a respondent had received the seasonal vaccine). We then used the top five results to define a new set of features, reducing the number of features from 54 to 5. 

Interestingly, the accurcy score of the new model was 74.5%, only slightly lower than our baseline model. This suggests that the additional features may be correlated with the other features, or irrelevant to the target variable. One advantage of using only a subset of the feautres is that the model was simpler and faster to train and predict. 


### ii. Hyperparameters tuning with RandomizedSearchCV
RandomizedSearchCV is a library from SKLearn that allows a user to perform hyperparameter tuning on a given model by specifying a list of hyperparameters to tune and a list of possible values for each. It then randomly selects a combination of hyperparameters from these lists and fits the model using them. The fit model is then scored using cross-validation, and the process is repeated a specified number of times. At the end, the model which resuted in the highest mean score across the cross-validation folds is selected as the best model, and the best comibation of hyperparameters is retained. 
We chose to use the following hyperparameters:
- n_estimators (refers to the number of trees in the forest);
- max_features (refers to the maximum number of features the model considers when looking for the best split at each tree node);
- min_samples_split (refers to the minimum number of samples required at a node in order for the node to be split).

We set up our search to train 20 models over 2 folds of cross-validation (resulting in 40 models total), scoring the best fit based on accuracy. The resulting hyperparameters were:
- n_estimators: 441;
- max_features: 14;
- min_samples_split: 19;
Resulting in 76.15% accuracy score, an improvement on our baseline model. 


### iii. Hyperparameters tuning with GridSearchCV
GridSearchCV is another library from SKLearn that allows us to tune the hyperparameters of a random forest model. It differs from RandomizedSearchCV in that it comprehensively searches over a specified hyperparameter grid, rathen than selecting a random number from the specified ranges of hyperparameters given.

To use GridSearch, we specified a grid of hyperparameter values, and the algorithm will train and evaluate a model for each combination of values. We used the following hyperparameters:
- n_estimators: 200, 500;
- max_features: 10, 15, 20;
- min_samples_split: 20, 25, 30.

Just like the RandomizedSearchCV library, the algorithm then looks at the model with the highest mean score across the cross-validation folds which is then selected as the best model, and the best combination of hyperparameters is retained. 

One advantage of GridSearchCV is that it is guaranteed to find the optimal combination of hyperparameters, since it examines all possible combinations. However, it can be very resource intensive since it requires examining every combination of hyperparameters. 

In the end, this method achieved 75.9% accuracy, a slight improvement on our baseline model.


### Evaluate the performance of each model
In this project, we built a baseline random forest machine learning model and three additional models based on feature selection, RandomizedSearchCV and GridSearchCV. The goal of the project was to evaluate the impact of these techniques on the model's performance, as measured by accuracy score. 

After building all the model's, we discovered that all four achieved similar accuracy scores. This suggests that the techniques we used to tune the model did not significantly improve the model's performance on the dataset. 

One possible explanation for this is that the baseline model was already well-tuned and didn't have much room to improve. On the other hand, it could also be that the hyperparameters we selected in our tuning methods were not optimal for this dataset. 

Future work could include trying different model configurations or algorithms, and looking at different feature subsets to train the model on. It would also be useful to evaluate the models using additional performance metrics, such as precision, recall, and F1 score, to get a more complete understanding of the models' performance. 


### An alternate: Deep Learning Model



*future development needed?*

## Conclusion


## Appendix 1
The table below describes the features included in the dataset.

 |Feature name                  | Description                                                                 |
 |------------------------------|-----------------------------------------------------------------------------|
 |behavioural_antiviral_meds    | Has taken antiviral medications. (binary)                                   |
 |behavioural_avoidance         | Has avoided close contact with others with flu-like symptoms. (binary)      |
 |behavioural_face_mask         | Has bought a face mask. (binary)                                            |
 |behavioural_wash_hands        | Has frequently washed hands or used hand sanitizer. (binary)                |
 |behavioural_large_gatherings  | Has reduced time at large gatherings. (binary)                              |
 |behavioural_outside_home      | Has reduced contact with people outside of own household. (binary)          |
 |behavioural_touch_face        | Has avoided touching eyes, nose, or mouth. (binary)                         |
 |doctor_recc_seasonal          | Seasonal flu vaccine was recommended by doctor. (binary)                    |
 |chronic_med_condition         | Has any of the following chronic medical conditions: asthma or an other lung condition, diabetes, a heart condition, a kidney condition, sickle cell anemia or another anemia, a neurological or neuromuscular condition, a liver condition, or a weakened immune system caused by a chronice illness or by medicines taken for a chronic illness. (binary)                                                                                    |
 |child_under_6_months          | Has regular close contact with a child under the age of six months. (binary)|
 |health_worker                 | Is a healthcare worker. (binary)                                            |
 |health_insurance              | Has health insurance. (binary)                                              |
 |opinion_seas_vacc_effective   | Respondent's opinion about seasonal flu vaccine effectiveness. <br> 1 = Not at all effective; <br> 2 = Not very effective; <br> 3 = Don't know; <br> 4 = Somewhat effective; <br> 5 = Very effective |
 |opinion_seas_risk             | Respondent's opinion about risk of getting sick with seasonal flu without vaccine. <br> 1 = Very low; <br> 2 = Somewhat low; <br> 3 = Don't know; <br> 4 = Somewhat high; <br> 5 = Very high          |
 |opinion_seas_sick_from_vacc   | Respondent's worry of getting sick from taking seasonal flu vaccine. <br> 1 = Not at all worried; <br> 2 = Not very worried; <br> 3 = Don't know; <br> 4 = Somewhat worried; <br> 5 = Very worried |
 |age_group                     | Age group of respondent.                                                     |
 |race                          | Race of respondent.                                                          |
 |sex                           | Sex of respondent.                                                           |
 |income_poverty                | Household annual income of respondent with respect to 2008 Census poverty thresholds.|
 |marital_status                | Marital status of respondent.                                                |
 |rent_or_own                   | Housing situation of respondent.                                             |
 |employment_status             | Employment status of respondent.                                             |
 |hhs_geo_region                | Respondent's residence using a 10-region geographic classification defined by the U.S. Dept. of Health and Human Services. Values are represented as short random character strings.                 |
 |census_msa                    | Respondent's residence within metropolitan statistical areas (MSA) as defined by the U.S. Census|
 |household_adults              | Number of other adults in household, top-coded to 3.                         |
 |household_children            | Number of children in household, top-coded to 3.                             |
 |employment_industry           | Type of industry respondent is employed in. Values are represented as short random character strings.|
 |employment_occupation         | Type of occupation of respondent. Values are represented as short random character strings.|

## References
* https://www.drivendata.org/competitions/66/flu-shot-learning/