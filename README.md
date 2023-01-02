#  Seasonal Flu Vaccination
*Prepared By Kit N., Lucas F. and Navpreet N.* <br> *12 January 2022*

## Introduction

This exercise is to predict whether individuals will receive their seasonal flu vaccine. To create the predictive model, we fit machine learning model to the vaccination data collected in the United States National 2009 H1N1 Flu Survey (NFHS)[^1]. Three machine learning models are considered:

* Model 1: Support vector classifier learning model
* Model 2: Random forest classsifier learning model
* Model 3: Deep neutral network model

We then compare the performance of each model to determine which has sufficient predictive power. *To add - discuss performance metric and what will indicate stronger performance*

[^1]: The NFHS was one-time survey designed to monitor vaccination during 2009-2010 flu season in response to the 2009 H1N1 pandemic. For further information, refer to [U.S. National 2009 H1N1 Flu Survey (NFHS)](https://webarchive.loc.gov/all/20140511031000/http://www.cdc.gov/nchs/nis/about_nis.htm#h1n1)


## The Dataset
 The NFHS data contains 26,707 survey respondents collected between late 2009 amd early 2010. Each survey respondents were asked whether they had received H1N1 and seasonal flu vaccines and additional questions about themselves such as social, economic and demographic background, opinions on risks of illness and vaccine effectiveness, and behaviours towards mitigating tranmission. 

 This dataset is a copy shared by the United States National Center for Health Statistics for a data science practice competition hosted by [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/).


### Data checking and cleaning
 Given the focus of this exercise is on seasonal flu vaccination prediction, H1N1 specific data included in the NFHS data is dropped and not used for fitting machine learning model. After removing H1N1 specific variables, the starting point is a dataset with 30 columns. The first column "respondent_id" is a unique identifier.


### Data exploration


## Machine Learning Model


*future development needed?*

## Conclusion



## References
* https://www.drivendata.org/competitions/66/flu-shot-learning/