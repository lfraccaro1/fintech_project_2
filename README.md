#  Seasonal Flu Vaccination
*Prepared By Kit N., Lucas F. and Navpreet N.* <br> *12 January 2022*

## Introduction

This exercise aims to predict whether individuals will receive their seasonal flu vaccine. To create the predictive model, we fit a machine learning model with random forest classifier (the baseline) to the vaccination data collected in the United States National 2009 H1N1 Flu Survey (NFHS)[^1]. 

Once the baseline is established, we explore various tuning techniques including:
* feature selection to reduce the number of input features,
* hyperparameters tuning with RandomisedSearchCV, and
* hyperparametrs tuning with GridSearchCV.

A second machine learning model involving deep learning is also considered. 

We then compare the performance of each model. The focused performance metric is accuracy score. This metric measures the number of correct predictions made by a model in relation to the total number of predictions made. This metric is calculated using sklearn.metrics.accuracy_score.

[^1]: The NFHS was one-time survey designed to monitor vaccination during 2009-2010 flu season in response to the 2009 H1N1 pandemic. For further information, refer to [U.S. National 2009 H1N1 Flu Survey (NFHS)](https://webarchive.loc.gov/all/20140511031000/http://www.cdc.gov/nchs/nis/about_nis.htm#h1n1)


## The Dataset
 The NFHS data contains 26,707 survey respondents collected between late 2009 and early 2010. Each survey respondents were asked whether they had received H1N1 and seasonal flu vaccines and additional questions about themselves such as social, economic and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviours towards mitigating tranmission. 

 This dataset is a copy shared by the United States National Center for Health Statistics for a data science practice competition hosted by [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/).

 Given the focus of this exercise is on seasonal flu vaccination prediction, H1N1 specific data included in the NFHS data is dropped and not used for fitting machine learning model. After removing H1N1 specific variables, the starting point is a dataset with 31 columns. The first column "respondent_id" is a unique identifier. The remaining columns include answer to additional questions asked in the survey. [Appendix 1](#appendix-1) describes the features in detail. <br>
 
### Data checking and cleaning
The following summarises the data checking and cleaning performed:
* **Class balance** - We check if there is any imbalanced class issue in the dataset. <br>
<img src="./Diagram/chk_balance_class.jpg" alt="drawing" width="280" height = "100"/> <br>
"seasonal_vaccine" target variable has balanced class.

* **Data quality** - We check if there is any missing or null value, using the ".isnull.sum() syntax. Out of the 31 columns: <br> - 6 columns (including the target variable column) has no missing value; <br> - 3 feature columns have more than 40% missing values; and <br> - the remaining feature columns have low volume of missing values where the missing value percentage ranges from 0.1% up to 17.0%. <br> <br> **Strategy for handling missing values** <br> - 3 features columns with more than 40% missing values are excluded from the modelling. These features are "health_insurance", "employment_industry" and "employment_occupation". <br> - For the remaining feature columns, missing values are replaced with the "most_frequent" value observed using sklearn.impute.SimpleImputer module. The most_frequent values are selected as it works well for both numerical and categorical variables.

* **Data type** - We review the data type using the ".info" function. Slightly more than half of the features are numerical variables. We encode the categorical variables into numerical values using sklearn.preprocessing.OneHotEncoder.

### Exploring the data
We study the vaccination pattern by plotting each single feature against the target variable. If a feature is correlated with the target, we expect to see different vaccination pattern as the values of the feature vary. 

Below are samples of the plot used to study the vaccination pattern. Opinion questions seem to have high correlation with the target, but not sex.

<img src="./Diagram/vacc_pattern.jpg" alt="drawing" width="750" height = "380"/>


## Build and Train Machine Learning Model
### Establish the baseline model
We decided to use the random forest machine learning model as our baseline for several reasons. First, random forests are known for their high accuracy and ability to handle large and complex datasets. This was important for us as we were working with a dataset that has a high number of features.

Second, random forests are robust to overfitting. This means the models are less likely to produce poor generalisation performance when applied to new data. This was an important consideration for our project as we wanted to ensure the model would be able to generalise well to new examples and not just perform well on the training data.

Last, random forest models are relatively simple to implement and do not require much fine-tuning, which we decided made it a good choice for a baseline model.

**Performance of the baseline model** <br>
<img src="./Diagram/class_rpt_baseline.jpg" alt="drawing" width="300" height = "130"/> <br>
The baseline model has an accuracy score of 76.13%.


### Tune the baseline
In this section, we explore both tuning parameters and tuning hyperparameters of the model.

#### i. Features Selection

Features selection involves reducing the number of input features used to train the model. To select which features to include, a [correlation matrix](./Diagram/corr.jpg) is computed to identify the features that are most correlated to the target variable. Four features with the highest absolute correlation value are used to define a new set of features, reducing the number of input features from 31 to 4. The new set of features are ["age_group_65+ years","doctor_recc_seasonal", "opinion_seas_risk", "opinion_seas_vacc_effective"]. 

<img src="./Diagram/class_rpt_lessfeat.jpg" alt="drawing" width="300" height = "130"/> <br>

The accuracy score of the model with new set of features was 75.15%, slightly lower than the baseline. This suggests the selected four features are relative strong signals for the target variable. Additional features may be correlated with these four features, or irrelevant to the target variable.


#### ii. Hyperparameters tuning with RandomizedSearchCV
RandomizedSearchCV is a library from SKLearn that allows a user to perform hyperparameter tuning on a given model by specifying a list of hyperparameter to tune and a list of possible values for each. It randomly selects a combination of hyperparameters from these lists and fits the model using them. The fitted model is then scored using cross-validation, and the process is repeated a number of times as defined by the user. The model that results in the highest mean score across the cross-validation folds is selected as the best model, and the best combination of hyperparameters is retained. 

We explored tuning the following hyperparameters:
* n_estimators (refers to the number of tress in the forest);
* max_features (refers to the maximum number of features the model considers when looking for the best split at each tree node);
* min_samples_split (refers to the minimum number of samples required at a node in order for the node to be split)

We set up the search to train 20 models over 2 folds of cross-validation (resulting in fitting 40 models total), scoring the best fit based on accuracy. 

<img src="./Diagram/class_rpt_rscv.jpg" alt="drawing" width="300" height = "130"/> <br>

The best fit model resulted in 77.18% accuracy score. The required hyperparameters are:
* n_estimators = 207,
* max_features = 10;
* min_samples_split = 23. <br>

This is an improvement on the baseline model.

#### iii. Hyperparameters tuning with GridSearchCV
GridSearchCV is another library from SKLearn that allows a user to perform hyperparameter tuning. It differs from RandomizedSearchCV in that it comprehensively searches over a specified hyperparameter grid, rather than randomly selecting a number from the specified ranges of hyperparamters given. One advantage of GridSearchCV is that it is guaranteed to find an optimal combination of hyperparameters, since it examines all possible combinations. However, it can be resource intensive since it examines every combination of hyperparameters.

Implementing tuning with GridSearchCV, we have used the following hyperparameters:
* n_estimators = 200, 500;
* max_features = 10, 15, 20;
* min_sample_split = 20, 25, 30.

The algorithm looks at the model with highest mean score across the cross-validation folds and selects it as the best model, and the combination of hyperparameters is retained. 

<img src="./Diagram/class_rpt_gscv.jpg" alt="drawing" width="300" height = "130"/> <br>

The GridSearchCV best model achieved 77.28% accuracy. The required hyperparameters are:
* n_estimators = 500,
* max_features = 15,
* min_samples_split = 30. <br>

This is an improvement on the baseline model.

### An alternate: Deep Learning Model
A second machine learning model involving binary classification using a neural network is considered. The neural network is two-layers deep and uses the relu activation function on both layers. This neural network model is compiled and fitted using the binary_crossentropy loss function, the adam optimiser, the accuracy evaluation metrics, 50 epochs and 1000 batch size. The neural network model summary is provided in the table below. <br>
<img src="./Diagram/nn_model_summary.jpg" alt="drawing" width="350" height = "200"/> <br>


**Neural network model performance**
<img src="./Diagram/nn_performance.jpg" alt="drawing" width="600" height = "230"/> <br>
As the epoch increases, loss decreases from 0.68 down to c.0.45 and accuracy increases from 0.58 up to c.0.75. The difference between train_loss and validation_loss is widening as epoch increases (similar observation made in metric accuracy). This could be a sign of overfitting. 


<img src="./Diagram/class_rpt_nn.jpg" alt="drawing" width="300" height = "130"/> <br>
The neural network model has an accuracy of 76.35%.

#### Tuning the Parameters to Find the Ideal Neural Network

The following hyperparameters were manipulated:<br>
* Input Features: 54<br>
* Output Neuron(s): 1<br>
* Loss: Binary Crossentropy<br>
* Optimizer: SGD<br>
* Metrics: Accuracy<br>
* Epochs: 150<br>
> *Layer 1*<br>
* Hidden Nodes: 27<br>
* Activation Function: Sigmoid<br>
* ropout Rate: 0.5<br>

> *Layer 2*<br>
* Hidden Nodes: 14<br>
* Activation Function: Sigmoid<br>
* Dropout Rate: 0.2<br>

> *Layer 3*<br>
* Hidden Nodes: 7<br>
* Activation Function: Sigmoid<br>
* Dropout Rate: 0.2<br>

> *Layer 4*<br>
* Hidden Nodes: 4<br>
* Activation Function: Sigmoid<br>

> *Output Layer*<br>
* Activation Function: Sigmoid

The tuned neural network model summary is provided in the table below. <br>
<img src="./Diagram/tuned_nn_model_summary.PNG" alt="drawing" width="350" height = "400"/> <br>

**Changes from Baseline:**<br>
Optimizer, Adam --> SGD;<br>
Activation functions in layers 1 and 2, ReLu --> Sigmoid;<br>
Addition of hidden layer 3, containing 7 neurons & Sigmoid activation function;<br>
Addition of hidden layer 4, containing 4 neurons & Sigmoid activation function; <br>
Epochs, 50 --> 150<br>
Addition of dropout technique to prevent overfitting, with rate of 0.5 for input layer and 0.2 for layers 2 and 3 (no dropout for layer 4).<br>
**Result:**<br>
Marginally higher AUC score than the baseline. Epochs were increased from 50 to 150 as it was found that while using the 'Dropout'regularization technique, the accuracy score following each epoch was increasing at a more reduced rate than before and thus a greater number of epochs is required to reach a sufficiently high score.

**Tuned neural network model performance**
The tuned model resulted in a marginally improved accuracy score of 76.28% as shown below:
<img src="./Diagram/tuned_nn_model_performance.PNG" alt="drawing" width="300" height = "150"/> <br>


### Comparing the performance of each model

| Model | Accuracy |
|-------|----------|
|**Baseline** : Random forest learning model uisng the full set of features | 76.13% |
|**Tune i** : Reduced number of input features | 75.15%|
|**Tune ii** : Hyperparameters tuning with RandomizedSearchCV | 77.18% |
|**Tune iii** : Hyperparameters tuning with GridSearchCV | 77.28% |
|**2nd ML** : Deep learning model | 76.35% |

* **Reduced number of input features** - Limiting the number of features resulted in a slight decrease in accuracy but it resulted in a simpler and faster model. This method will be useful in cases where computational resources are limited or when interpretability of the model is as priority.

* **Hyperparameters tuning with RandomizedSearchCV** - It improves the accuracy and is less resource intensive compared to GridSearchCV. RandomizedSearchCV is recommended when there is a large parameter space tuning required.

* **Hyperparameters tuning with GridSearchCV** - While GridSearchCV resulted in the highest accuracy score, it is the most resource intensive approach due to its comprehensive search of all possible hyperparameter combinations. 

* **Deep learning model** - The simple neural network model has similar accuracy as the baseline and is a good alternative. 


## Future Work
Future work could include:
* explore different model configurations and/or algorithms;
* explore different features sets using feature engineering technique to train the model;
* evaluate the models using additional performance metrics such as precision, recall and F1 score to get a more comprehensive understanding of the model's performance.
* further tuning the neural network model such as introducing more layers, dropout regularisation and tuning learning rate.

## Conclusion
The baseline model has a reasonably high accuracy score of 76.13%. Both attempts on hyperparameters tuning only improve the accuracy marginally. This could be due to the baseline model being already well-tuned and did not have much room to improve. It could also be that the specified hyperparameters ranges or values did not capture the optimal set.

The chosen simple neural network has similar predictive power as the baseline model. Further tuning may increase the accuracy and outperform the baseline model.

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

## Appendix 2 - Correlation Heatmap

<img src="./Diagram/corr.jpg" alt="drawing" width="850" height = "650"/> 

## References
* Data source: https://www.drivendata.org/competitions/66/flu-shot-learning/
* SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
* RandomizedSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
* GridSearchCV : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
* Statistical Function (scipy.stats): https://docs.scipy.org/doc/scipy/reference/stats.html
