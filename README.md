# Titanic Classification Project

This project aims to predict the survival of passengers on the Titanic, based on their personal and ticket information. The project uses a labeled dataset from Kaggle, which contains features such as age, gender, class, fare, cabin, and embarked port. The project applies various machine learning algorithms, such as logistic regression, decision tree, random forest, and support vector machine, to train and evaluate different models. The goal is to create a model that can accurately classify new data, reflecting the likelihood of survival in a Titanic-like scenario.
## Data Analysis

The project performs some basic data analysis, such as descriptive statistics, histograms, correlation heatmap, and pivot tables, to explore the data and draw some insights. Some of the findings are:

- The average age of survivors is 28, so young people tend to survive more.
- People who paid higher fare rates were more likely to survive, more than double. This might be the people traveling in first-class. Thus the rich survived, which is kind of a sad story in this scenario.
- If you have parents, you had a higher chance of surviving. So the parents might’ve saved the kids before themselves, thus explaining the rates.
- Women had a higher chance of survival than men, which might be due to the "women and children first" policy.

## Feature Engineering

The project performs some feature engineering, such as imputing missing values, creating new features, encoding categorical variables, and scaling numerical variables, to prepare the data for modeling. Some of the steps are:

- Imputing the missing values in the Age column with the median age of the corresponding Pclass group.
- Creating a new feature called Title, which extracts the title from the Name column, such as Mr, Mrs, Miss, etc.
- Creating a new feature called FamilySize, which combines the SibSp and Parch columns, and indicates the total number of family members on board.
- Encoding the categorical variables, such as Sex, Title, Cabin, and Embarked, with one-hot encoding or label encoding, depending on the number of unique values.
- Scaling the numerical variables, such as Age, Fare, and FamilySize, with standardization or normalization, to make them comparable and reduce the effect of outliers.

## Modeling

The project applies four different machine learning algorithms to train and evaluate the models, using the training data and cross-validation. The algorithms are:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine

The project uses accuracy as the main metric to compare the performance of the models, as well as other metrics such as precision, recall, and f1-score. The project also plots the confusion matrix and the ROC curve for each model, to visualize the trade-off between true positive rate and false positive rate.

The project selects the best model based on the highest accuracy score on the validation data, and applies it to the test data to generate the predictions. The project then submits the predictions to the Kaggle competition, and obtains the final score on the test data.

## Conclusion

The project demonstrates how to apply various machine learning techniques to a classification problem, using the Titanic dataset as an example. The project shows how to perform data analysis, feature engineering, and modeling, and how to evaluate and compare different models. The project also shows how to generate and submit predictions to a Kaggle competition, and how to interpret the results.

The project can be further improved by trying more advanced algorithms, such as neural networks or gradient boosting, or by tuning the hyperparameters of the existing algorithms, using grid search or random search. The project can also be extended by adding more features, such as the name length, the ticket prefix, or the cabin deck, or by applying more sophisticated feature engineering techniques, such as feature selection, feature extraction, or feature interaction.
