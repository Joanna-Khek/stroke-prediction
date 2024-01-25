# Stroke Prediction

## Project Description
This project aims to create a model that can accurately predict whether a user is likely to develop stroke, based on characteristics such as ``gender``, ``age``, ``hypertension``, ``heart_disease``, ``ever_married``, ``work_type``, ``Residence_type``, ``avg_glucose_level``, ``smoking_status`` and ``bmi``.     

The data contains 15304 rows and 11 features.      

Tools such as **Hyperopt** and **MLflow** were utilised for hyperparamter tuning. **PyTest** was utilised for unit tests.

## Exploratory Data Analysis
The full EDA notebook can be found in ``notebooks/eda.ipynb``. Here are some of the key insights.

- Based on the ``age`` variable, older people are more susceptible to stroke
- Based on the ``avg_glucose_level`` and ``bmi`` variable, there is not much difference in the distribution between people who has stroke and no stroke
![continuous](https://github.com/Joanna-Khek/stroke-prediction/blob/main/assets/cont_boxplot.png)

- People with hypertension are more likely to develop stroke

![hypertension](https://github.com/Joanna-Khek/stroke-prediction/blob/main/assets/discrete_Hypertension_prop.png)

- People with heart disease are more likely to develop stroke
![heart_disease](https://github.com/Joanna-Khek/stroke-prediction/blob/main/assets/discrete_Heart%20Disease_prop.png)

  
## Model Results
The evaluation metric is the AUC-ROC.

A baseline result of 5-fold cross validation was obtained for the following models:
| Model  | Training  | Validation |
| ------------- | ------------- | ------------- |
| Logistic Regression  | 0.8802 | 0.8773
| Decision Tree  | 1.0  | 0.5861
| Random Forest  | 1.0  | 0.8225
| XGBoost  | 0.9981  | 0.8504
| Gradient Boosting  | 0.9196 | 0.8794

- There seem to be overfitting for decision tree and random forest. Logistic regression surprisingly performs well.

- Performing hyperparamter tuning on the best model - Gradient Boosting, we see that in general, the average validation ROC decreases as the learning rate increases and when the number of trees increases.

![hyper](https://github.com/Joanna-Khek/stroke-prediction/blob/main/assets/hyperparams_tuning.png)


