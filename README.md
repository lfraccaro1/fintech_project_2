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

 Given the focus of this exercise is on seasonal flu vaccination prediction, H1N1 specific data included in the NFHS data is dropped and not used for fitting machine learning model. After removing H1N1 specific variables, the starting point is a dataset with 30 columns. The first column "respondent_id" is a unique identifier. The table below describes the remaining columns: <br>
 
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

### Data checking and cleaning


### Data exploration


## Machine Learning Model


*future development needed?*

## Conclusion



## References
* https://www.drivendata.org/competitions/66/flu-shot-learning/