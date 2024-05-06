# Customer Survival Analysis and Churn Prediction

Customer attrition, also referred to as customer churn, turnover, or defection, involves the loss of clients or customers.

Telecommunication providers, Internet service companies, pay TV providers, insurance companies, and alarm monitoring services frequently utilize customer attrition analysis and rates as critical business metrics. This is because retaining an existing customer typically costs significantly less than acquiring a new one. These industries often operate customer service departments dedicated to reclaiming customers who have left, as re-engaged long-term customers can be considerably more valuable than new ones.

Predictive analytics deploy churn prediction models to forecast customer defection by evaluating their likelihood of churn. These models create a prioritized list of potential defectors, which helps to effectively target customer retention efforts towards those most likely to leave.

Through this project, I intend to conduct customer survival analysis and develop a model that can predict customer churn. Additionally, I plan to create an application that helps to decipher why specific customers may discontinue service and assess their projected lifetime value.

## Final Customer Churn Prediction App
<img src=app-pic.png>

## Project Structure
'''
.
├── Images/                             : contains images
├── static/                             : plots to show gauge chart, hazard and survival curve, shap values in Flask App 
│   └── images/
│       ├── hazard.png
│       ├── surv.png
│       ├── shap.png
│       └── new_plot.png
├── templates/                          : contains html template for flask app
│   └── index.html
├── Customer Survival Analysis.ipynb    : Survival Analysis kaplan-Meier curve, log-rank test and Cox-proportional Hazard model
├── Exploratory Data Analysis.ipynb     : Data Analysis to understand customer data
├── Churn Prediction Model.ipynb        : Random Forest model to predict customer churn
├── app.py                              : Flask App
├── app-pic.png                         : Final App image  
├── explainer.bz2                       : Shap Explainer
├── model.pkl                           : Random Forest model
├── survivemodel.pkl                    : Cox-proportional Hazard model
├── requirements.txt                    : requirements to run this model
├── Procfile                            : procfile for app deployment
├── LICENSE.md                          : MIT License
└── README.md                           : Report
'''


## Customer Survival Analysis

Survival analysis encompasses a range of techniques used to examine data in which the primary outcome is the duration until a particular event occurs. This event could be as varied as death, the onset of a disease, marriage, or divorce, and the duration until the event—referred to as survival time—can span days, weeks, or years.

For instance, if considering heart attacks, the survival time might be measured in years from a starting point until the individual experiences a heart attack.

**Objective:**
The aim of this study is to apply non-parametric and semi-parametric survival analysis methods to explore several key questions:

- How does the probability of customer churn evolve over time?
- How can we establish a model that links customer churn to time and other relevant characteristics of the customer?
- Which factors are most influential in driving customer churn?
- What are the survival and hazard curves for a particular customer?
- How can we calculate the expected lifetime value of a customer?

**Kaplan-Meier Survival Curve:**

<p align="center">
<img src="/Images/SurvivalCurve.png" width="400" height="300">
</p>

From the graph mentioned, it is observable that:

- As anticipated, in the telecom industry, churn rates are comparatively low, with the company maintaining over 60% of its customers for more than 72 months.
- There is a steady decline in the survival probability from 3 to 60 months.
- Beyond 60 months, or after five years, the rate of decline in survival probability accelerates significantly.

**Log-Rank Test:** 

This test helps to compare survival distributions of different groups. Below are the survival curves segmented by various customer attributes.

<p align="center">
<img src="/Images/gender.png" width="250" height="200"/> 
<img src="/Images/Senior%20Citizen.png" width="250" height="200"/>
<img src="/Images/partner_1.png" width="250" height="200"/> 
</p>

**Survival Regression:**
I utilized the Cox-proportional hazard model for survival regression, which is ideal for examining the impact of various risk factors on survival time.

<p align="center">
<img src="/Images/gender.png" width="250" height="200"/> 
<img src="/Images/Senior%20Citizen.png" width="250" height="200"/>
<img src="/Images/partner_1.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="/Images/dependents.png" width="250" height="200"/> 
<img src="/Images/phoneservice.png" width="250" height="200"/>
<img src="/Images/MultipleLines.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="/Images/InternetService.png" width="250" height="200"/> 
<img src="/Images/OnlineSecurity.png" width="250" height="200"/> 
<img src="/Images/OnlineBackup.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="/Images/DeviceProtection.png" width="250" height="200"/> 
<img src="/Images/TechSupport.png" width="250" height="200"/>
<img src="/Images/Contract.png" width="250" height="200"/> 
</p>

<p align="center">
<img src="/Images/StreamingMovies.png" width="250" height="200"/>
<img src="/Images/paymentmethod.png" width="250" height="200"/> 
<img src="/Images/PaperlessBilling.png" width="250" height="200"/>
</p>

From the graphs provided, we can draw the following conclusions:

- The customer's gender and the type of phone service do not appear to be significant predictors of churn, as indicated by their p-values in the log-rank test, which are above the threshold of 0.05.
- Younger customers with families tend to have lower churn rates, possibly due to factors such as a busier lifestyle and higher income.
- Customers who do not subscribe to additional services such as online backup, online security, device protection, tech support, streaming TV, and streaming movies—despite having active internet service—show a reduced survival probability.
- The company should focus on customers who subscribe to internet services since their likelihood of churning is consistently higher. Particularly, the Fiber Optic type of internet service, which is more expensive and faster than DSL, may contribute to a higher churn rate.
- The company should offer more incentives to customers on month-to-month contracts to encourage them to commit to long-term services.
- Customers who use automatic payment methods are less likely to churn compared to those who pay via electronic checks or mailed checks, as the latter requires more effort and time to process payments.

**Survival Regression:**
I employ the Cox proportional hazard model to conduct survival regression analysis on customer data. This model effectively links multiple risk factors or exposures to the survival time. In the Cox proportional hazards regression model, the effect size is represented by the hazard rate, which quantifies the risk or probability of the event of interest occurring, provided that the participant has survived up to a certain time. The model provides a good fit to the data, and the coefficients are detailed below.

<p align="center">
<img src="/Images/Survival-analysis.png" width="750" height="500"/>
</p>

By applying this model, we can derive the survival and hazard curves for any customer. These curves are instrumental in estimating the remaining lifespan of a customer.

<p align="center">
<img src="/Images/survival.png" width="400" height="300"/>
<img src="/Images/hazard.png" width="400" height="300"/>
</p>

**Customer Lifetime Value:**
To estimate a customer's lifetime value, I calculate it by multiplying the monthly charges the customer pays to Telecom by their expected lifetime.

I use the survival function to estimate a customer's expected lifetime. I adopt a conservative approach, assuming a customer has churned when their survival probability drops to 10%.

## Customer Churn Prediction
My goal is to develop a machine learning model that can precisely predict whether a customer will churn.

###Analysis

**Churn and Tenure Relationship:**

<p align="center">
<img src="https:/Images/tenure-churn.png" width="600" height="300"/>
</p>

- It's evident that the longer the tenure, the lower the churn rate, indicating increased customer loyalty over time.

<br />

**Tenure Distrbution by Various Services:**

<p align="center">
<img src="/Images/tenure-dist.png" width="340" height="250"/>
</p>

- Newly acquired customers often do not subscribe to various services, and their churn rate is significantly high, as observed in the graph for Streaming Movies. This pattern holds for other services as well.

<br />

**Internet Service By Contract Type:**

<p align="center">
<img src="/Images/internetservice-contract.png" width="360" height="250"/>
</p>

- A substantial number of customers who choose a month-to-month contract opt for Fiber optic internet service, which correlates with a higher churn rate for this type of service.

<br />

**Payment method By Contract Type:**

<p align="center">
<img src="/Images/payment-contract.png" width="500" height="250"/>
</p>

- Customers on month-to-month contracts tend to pay mostly by Electronic Check or mailed check, possibly due to the shorter cancellation process compared to automatic payments.

<br />

**Monthly Charges:**

<p align="center">
<img src="/Images/monthlycharges.png" width="300" height="220"/>
</p>

- It is noticeable that customers who pay higher monthly fees tend to churn more.

<br />

### Modelling

For the modeling process, I will employ a tree-based ensemble method, as the classification issue at hand lacks linearity. Additionally, we are dealing with a class imbalance ratio of 1:3. To address this, I will assign a class weighting of 1:3, implying that false negatives are considered three times more detrimental than false positives. The model was trained on 80% of the dataset and validated on the remaining 20% to ensure there was no data leakage. The random forest model, rich with hyperparameters, was fine-tuned using Grid Search Cross Validation to prevent overfitting.

The final model achieved an F1 score of 0.62 and an ROC-AUC of 0.85. The charts resulting from the model are displayed below.

<p align="center">
<img src="/Images/model_1.png" width="600" height="300"/>
<img src="/Images/model_feat_imp.png" width="600" height="400"/>

</p>

From the feature importance graph, we can discern which attributes significantly influence customer churn.

### Explanation

We employ explainable AI tools such as Permutation Importance, Partial Dependence plots, and Shap values to interpret and understand the Random Forest model:

1. Permutation Importance - This technique assesses the importance of a feature by randomly shuffling its values and observing the impact on model performance.

<p align="center">
<img src=/Images/eli51.png height=250 width=200>
<img src=/Images/eli52.png height=130 width=200> 
</p>

2. Partial Dependence Plot - This plot helps visualize how changes in specific features affect the probability of churn. For instance, the graph below illustrates that for customers in tenure group 2, the likelihood of churn decreases more steeply compared to those in tenure group 1.

<p align="center">
<img src=/Images/pdp_tenure.png height=250 width=400>
<img src=/Images/pdp_contract.png height=250 width=400> 
</p>

<p align="center">
<img src=/Images/pdp_monthly_charges.png height=250 width=400>
<img src=/Images/pdp_total_charges.png height=250 width=400> 
</p>

3. Shap Values (SHapley Additive exPlanations) - This game-theoretic approach helps explain the output of machine learning models. The plot below clarifies why the churning probability of a particular customer is below the baseline and identifies the contributing features.

![](/Images/shap.png)

## Flask App

I have saved the finely tuned Random Forest model and deployed it using a Flask web app. Flask is a lightweight web framework designed for simplicity and scalability. I also stored the Shap value explainer configured with the Random Forest model to display Shap plots within the app. Additionally, I've implemented the Cox-proportional hazard model to demonstrate survival and hazard curves and to estimate the expected lifetime value of a customer.

The final application displays the probability of churn, a gauge chart indicating the severity of a customer’s potential churn, and Shap values based on individual customer data. The layout of the final app can be seen above.

<p align="center">
<img src="/static/app-pic.png" width="500" height="300"/>
</p>

