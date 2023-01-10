# Customer-Churning--Classification-Model
This is a banking data, where we will be predicting if the customer will churn in the next quarter or not.(Average balance of customer falls below minimum balance in the next quarter (1/0)) using EDA &amp; different Classification Algorithm
Steps to be follwed are:

* Load Data & Packages for model building & preprocessing
* Missing value imputation
* Feature Engineering
* Exploratory Data Analysis
* Preprocessing
* Select features on the basis of EDA Conclusions & finalise columns for model
* Decide Evaluation Metric on the basis of business problem
* Build model -Logistic,Random Forest,XGBoost using Cross Validation
* Hyperparameter Tuning using Grid Search CV
* Use Reverse Feature Elimination to find the top features and build model using the top features with accuaracy & compare

Libraries used:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit Learn

Final Result:
![image](https://user-images.githubusercontent.com/57029230/211523140-b0991ede-86de-452e-acf9-cb35a4b462b6.png)
array([[5584,  197],T (actual)
       [ 778,  537]]F
          T      F (predicted)
          
 Accuracy: 86%         
