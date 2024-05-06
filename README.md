# customer_churn_prediction

# Customer Survival Analysis and Churn Prediction

This project delves into survival analysis to monitor how customer churn probability evolves over time and computes the lifetime value (LTV) of customers. Additionally, it employs a Random Forest model to forecast potential customer churn and integrates the model within a Flask web application for practical use.

## Final Customer Churn Prediction App
<img src=https://github.com/anayathawale/customer_churn_prediction/app-pic.png>

## Project Structure
'''
├── Images/ : stores images
├── static/ : holds plots for gauge chart, hazard and survival curves, shap values in Flask App
│ └── images/
│ ├── hazard.png
│ ├── surv.png
│ ├── shap.png
│ └── new_plot.png
├── templates/ : contains HTML templates for Flask app
│ └── index.html
├── Customer Survival Analysis.ipynb : Notebook for Survival Analysis including Kaplan-Meier curve, log-rank test, and Cox-proportional Hazard model
├── Exploratory Data Analysis.ipynb : Notebook for preliminary data exploration
├── Churn Prediction Model.ipynb : Notebook for developing the Random Forest churn prediction model
├── app.py : Flask application script
├── app-pic.png : Snapshot of the final app interface
├── explainer.bz2 : Shap Explainer object
├── model.pkl : Serialized Random Forest model
├── survivemodel.pkl : Serialized Cox-proportional Hazard model
├── requirements.txt : Dependencies required for the project
├── Procfile : Configuration for app deployment
├── LICENSE.md : MIT License document
└── README.md : Project documentation
'''


## Customer Survival Analysis

**Survival Analysis:** 
This branch of statistics is focused on examining the time duration until an event occurs, such as customer churn. It's critical for understanding customer retention.

**Objective:**
The goal is to apply non-parametric and semi-parametric survival analysis methods to explore:
- Changes in churn likelihood over time
- Modeling the relationship between churn, time, and customer attributes
- Significant drivers of customer churn
- Individual customer survival and hazard functions
- Estimating a customer's lifetime value

**Kaplan-Meier Survival Curve:**

<p align="center">
<img src="https://github.com/anayathawale/customer_churn_prediction/Images/SurvivalCurve.png" width="400" height="300">
</p>

This curve illustrates:
- The telecom company retains over 60% of customers after 72 months.
- A consistent decline in survival probability between 3-60 months.
- A steeper decline in survival probability after 60 months.

**Log-Rank Test:** 

This test helps to compare survival distributions of different groups. Below are the survival curves segmented by various customer attributes.

<p align="center">
<img src="https://github.com/anayathawale/customer_churn_prediction/blob/master/Images/gender.png" width="250" height="200"/> 
<img src="https://github.com/anayathawale/customer_churn_prediction/blob/master/Images/Senior%20Citizen.png" width="250" height="200"/>
<img src="https://github.com/anayathawale/customer_churn_prediction/blob/master/Images/partner_1.png" width="250" height="200"/> 
</p>

**Survival Regression:**
We utilize the Cox-proportional hazard model for survival regression, which is ideal for examining the impact of various risk factors on survival time.

<p align="center">
<img src="https://github.com/anayathawale/customer_churn_prediction/blob/master/Images/Survival-analysis.png" width="750" height="500"/>
</p>

**Customer Lifetime Value:**
We compute this by multiplying the customer's monthly charges by their expected lifetime, based on survival functions.

## Flask App

The Flask app showcases the model's predictions and provides insights into factors influencing individual predictions, such as SHAP values.

<p align="center">
<img src="https://github.com/anayathawale/customer_churn_prediction/blob/master/static/app-pic.png" width="500" height="300"/>
</p>

