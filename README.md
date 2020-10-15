# IBM-HR-employee-Attrition-Prediction-Study
## Dataset source
- https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

*In this project, we perform a complete machine learning work flow from data exploratory and feature engineering to build different machine learning models and evaluate model performances to predict if an employee will leave his/her current employer and analyze what features will mainly affect employee's attrition activity.*

# Contents of the project
  
  1. Data Exporatory and analysis

  2. Feature engineering

  3. Model training and performance evaluations

  4. Feature importance analysis

## Data Exporatory and analysis

*Data exporatory is the most important part of the work flow for machine learning project as it is the first approach to understand the whole dataset and all the features including numerical and non numerical, missing data, duplicate data, meaningful and meaningless. Since we will try the best to understand and apply all the useful features to the models later on, we would have to make ourself understand all the features that we have.*

### Visualization
*It is one of the most effective ways to understand the statistical distribution and detect potential outliers in our dataset.*

#### Distribution of Target Variable
![Attrition Distribution](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/attrition.png)

As we could see the distribution of our target variable is **imbalanced** for the binary variable yes and no.

#### Distribution of Numerical Features
![Numerical Distribution](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/numerical_dist.png)

This is the distribution of our numerical features. We have preprocessed all the numerical features so there are no missing data or duplicate data. 
From the exproratory, it looks like exit employee's **age** and **working time with his current position** give out some signals of attrition as well as **dailyrate** which refers to paystub.

#### Distribution of Non Numerical Features
![Non Numerical Distribution](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/cate_dist.png)

This is the distribution of non numerical features. Even though all the distribution is imbalanced, we could still tell that too much **over time** work definitely will drive away a loyal employee. **Traveling** and **martial status** will also be considered as factors of the effect.

## Feature Engineering

#### Correlation Matrix
![Correlationn Matrix](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/corr.png)

From this matrix we could be able to see correlations between each feature and between features and target variable. For correlation score > 0.5, we would consider the two features are correlated.

*For non numerical features, we applied one hot encoding and transform them into numerical.*

## Model Training

#### Naive Approach
![Naive Approach](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/draft.png)

Naive approach is to apply features to the models without hyperparameter tuning. We could roughly get the result of performance of each model. Here we use the **linear model, tree based model, ensemble learning model and deep learning model**.

#### Hyperparameter Tuning

*Here are the graphs with different hyperparameter affect the performance of logistic regression and k nearest neighbors.*

![LR](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/LR.png)
![KNN](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/knn.png)

*K fold cross validation is the method we use to check the performance of the model on different dataset, so basically we split our dataset into trainig set and testing set, and we split training set into same different portions, and we apply each portion to our model and get the mean score of the model performance. Then we will apply use our testing set to verify the accuracy of our predictions.*


## Model Evaluation

*There are different metrics to evalutate a model's performance. For classification problem, which is what we tried to solve in this project, accuracy is one of the metric that we will look at. Beside accuracy, precision and recall is another metric that we need to pay more attention especially for imbalanced variable. Assuming we only have 1 employee exit in our dataset and we predict everyone is staying, then we will have a model with accuracy 99% which still cannot be able to help us to find out the employee who would like to exit. So precision recall and f1 score will the metric for our evaluation. And to better evaluate and visualize the result of precision and recell, we use confusion matrix graph with labels of acutal and prediction.*

![cm_lr](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/cm_lr.png)
![cm_knn](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/cm_knn.png)
![cm_rf](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/cm_rf.png)
![cm_dt](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/cm_dt.png)
![cm_mlp](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/cm_mlp.png)
![cm_xgb](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/cm_xgb.png)

*ROC analysis is another metric to evaluation how well the model could separate the true label and false lablel accurately.*
![roc](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/ROC.png)

AUC a.k.a area under curve, the higher the value is the less randomness the model will generate for the correct true or false label here is yes or no. The ideal AUC will be 1 which will fill up the whole square and worst is the triangle that goes diagnoal across the squre.

## Feature Importance

*This is the most important analysis that gained from random forest and LASSO aka linear model with regularization. The feature importance reveals how important each feature is and how it will affect the performance of the model.*

![rf_fi](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/feature_importance.png)
![LASSO_fi](https://github.com/whwbj/IBM-HR-employee-Attrition-Prediction-Study/blob/main/graphs/L1_feature_importance.png)

## Conclusions

*It is quite obvious and it is a conclusion that everyone will accept, which is paying more and working less may be the eltimate dream for all employees.
Overtime, monthlyrate, age, years of works they all are very important feature that will affect the attrition activity and performance of an employee. So it will be quite efficient to apprach to those higher risk employees and better communicate with them, so that their intention of attrition could be resolved. It will be beneficial for both employer and employee. Otherwise, plan ahead of time to hire new candidate will be an alternative option.*

## Future Plans and Works

*This is a complete flow for machine learning and there is still much of work to do. We will need to spend more time on tuning the model so that we will have a more robust model that could be deployed in the future. If possible, we could collect more data and more feaetures so that the model could learn more from data and make more accurate predictions.*
