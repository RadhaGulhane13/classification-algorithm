'
## Overview

The objectives of this assignment are:
- Solve a business problem by creating, evaluating, and comparing three classification models, and produce the outputs needed to provide business value for your stakeholders.
- Experiment with built-in classification models in scikit-learn.
Dataset

## Dataset
- For this project, dataset, "hotel_bookings_cleaned_dataset.csv" has been used.
T- This dataset was pulled on 4/8/22 from: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

- The data has been altered slightly for use in course assignments,etc.:
    - A unique ROW attribute has been added.
    - Errors have been added, such as: duplicated records, deleted records, deleted - attribute values, erroneous attribute values.
- The dataset "hotel_bookings_cleaned_dataset.csv" is the dataset obtained by applying data-processing techniques.

## Problem Statement
Assume that you are the Director of Data Science for Buckeye Resorts, Inc. (BRI), an international hotel chain. As is the case for all hotel chains, reservation cancellations cause significant impacts to BRI, in profitability, logistics, and other areas. Approximately 20% of reservations are cancelled, and the cost to BRI of a cancelled reservation is $500 on average.

BRI wants to improve (decrease) the cancellation rates at its hotels, using more tailored interventions, based on newly available detailed data. BRI processes 100,000 reservations per year, so an incremental improvement in cancellation rates would have a significant impact.

One intervention being considered is to offer a special financial incentive to customers who have reservations, but who are “at risk” of cancellation. BRI has performed a small pilot test, and has found that offering a $80 discount to a customer who is planning to cancel is effective 35% of the time in inducing the customer not to cancel (by locking in a “no cancellation” clause).

BRI leadership has asked your team to analyze the new data, and determine if it is suitable for developing analyses and models that would be effective in predicting which future reservations are likely to be at risk of cancellation, so the aforementioned financial incentive could be offered.

The head of BRI would then like you attend the upcoming BRI Board of Directors meeting. She has asked you to present your findings to her and to the BOD, to help them decide whether to go forward with the planned tailored intervention approach, and/or to adjust or abandon the approach. Your goal is to support the BOD in making a decision.

## Analysis

### Define measures that do not include the cost information

```
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def calculate_cost(conf_matrix):
    cost_matrix = np.array([[95, 0], [-80, 0]])
    
    total_cost = np.sum(conf_matrix * cost_matrix)

    return total_cost
```

###  Evaluation of the Off-The-Shelf KNN Classifier
```
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test) 
knn_cm = confusion_matrix(y_test, knn_pred)
knn_accuracy_score = accuracy_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_precision_score = precision_score(y_test, knn_pred)
knn_total_cost = calculate_cost(knn_cm)
```

#### General Characteristics of KNN

- KNN is generally good for small to medium-sized datasets.
- When the problem is multi-class classification and the classes are well-separated, KNN can perform well.
- KNN can also be effective in problems where the training data is highly imbalanced, as it can adapt to the minority class better than other algorithms.
- KNN can be computationally expensive and slow when working with large datasets or high-dimensional feature spaces.
When the decision boundary is linear or nearly linear, KNN may not perform well.
- KNN is sensitive to irrelevant features, so if the dataset has many irrelevant features, it may not be a good choice. KNN does not work well with missing values, so if the dataset has many missing values, KNN may not be a good choice.

#### Anaysis of given problem and data for KNN
- As we have unbalanced dataset , KNN can work well.
- On the other hand, as given dataset is very large, choosing KNN as a classifier model can cause computational resourses and training time. Moreover, as KNN is sensitive to irrelevant features and given dataset has irrelevant attributes where we are unaware of correct attribute choice, KNN may not give good performance.

### Evaluation of Off-The-Shelf Decision tree Classifier

``` 
decision_tree_clf = tree.DecisionTreeClassifier(max_depth=i, criterion='entropy', max_features='auto')
decision_tree_clf.fit(X_train, y_train)
decision_tree_pred = decision_tree_clf.predict(X_test) 
decision_tree_cm = confusion_matrix(y_test, decision_tree_pred)
decision_tree_accuracy_score = accuracy_score(y_test, decision_tree_pred)
decision_tree_recall = recall_score(y_test, decision_tree_pred)
decision_tree_precision_score = precision_score(y_test, decision_tree_pred)
decision_tree_total_cost = calculate_cost(decision_tree_cm)
```

#### General Characteristics of Decision Tree Classifier

- Decision trees are good when the data contains categorical variables or features with discrete values.
- Decision trees are also good when there are nonlinear relationships between the features and the target variable.
- Decision trees can easily overfit the data, particularly when the tree is deep and complex. This can lead to poor generalization and low accuracy on new data.
They may not perform well when the data has continuous features, as decision trees typically partition the feature space into regions with discrete boundaries.
- Decision trees can be sensitive to noise and outliers in the data, which can lead to suboptimal splits and poor accuracy. They can be computationally expensive and slow when working with large datasets or high-dimensional feature spaces.
Anaysis of given problem and data for Decision Tree Classifier

#### Anaysis of given problem and data for Decision Tree
- As we have categorical variables or discrete value features in given dataset , this classifier can perform well.
- On the other hand, as given dataset is very large, choosing Decision Tree Classifier as a classifier model can cause computational resourses and training time. Moreover, depth of the tree needs to be choosen carefully to avoid overfitting.


### Evaluation of Off-The-Shelf Random Forest Classifier

``` 
random_forest_clf = RandomForestClassifier(n_estimators=e, criterion='entropy', max_depth=d)
random_forest_clf.fit(X_train, y_train)
random_forest_pred = random_forest_clf.predict(X_test) 
random_forest_cm = confusion_matrix(y_test, random_forest_pred)
random_forest_accuracy_score = accuracy_score(y_test, random_forest_pred)
random_forest_recall = recall_score(y_test, random_forest_pred)
random_forest_precision_score = precision_score(y_test, random_forest_pred, zero_division=1)
random_forest_total_cost = calculate_cost(random_forest_cm)
```

#### General Characteristics of Random Forest Classifier

- Random Forest is effective in handling large datasets with many features, as it reduces overfitting by randomly selecting a subset of features and samples for each tree.
- It is good at handling missing data and maintaining high accuracy on imbalanced datasets.
- Random Forest can handle both categorical and continuous features, making it a versatile algorithm.
- Random Forest is computationally more expensive than a single decision tree and may take longer time to train and predict, especially when the number of trees in the ensemble is large.
- It can be difficult to interpret the results of a Random Forest, as it is a combination of many decision trees.

#### Anaysis of given problem and data for Decision Tree Classifier

- Random Forest Classifier can perform better for the given dataset as it deals with overfitting problem , works well with both categorical and continuous features and handle large and imbalanced dataset.
- However, it is computationally expensive. Thus, can take longer time to train the model.

## Results
| Sr.No  |  Classifier              | Accuracy |  Recall  |  Precision | Net Benefit |
|--------|--------------------------|----------|----------|------------|-------------|
| 0      | KNN_default              | 0.690909 | 0.353130 |  0.672783  | 55160       |
| 1      | KNN_with_3_neighbour     | 0.726061 | 0.537721 |  0.671343  | 58945.0     |
| 2      | Decision Tree Classifier | 0.763636 | 0.552167 |  0.756044  | 64700       |
| 3      | Random Forest Classifier | 0.762424 | 0.377207 |  0.983264  | 66145       |

## Conclusion
- From the results, we can see that the Decision Tree Classifier and Random Forest Classifier have similar accuracy scores, but the Decision Tree Classifier has a higher recall and precision score. The Random Forest Classifier, on the other hand, has the highest precision score but the lowest recall score. The KNN with 3 neighbors classifier has a lower accuracy score but a higher recall and precision score compared to the Random Forest Classifier. Based on the problem statement we want high recall than precision.
- In terms of net benefit, the Decision Tree Classifier has the highest score, followed closely by the Random Forest Classifier. The KNN with 3 neighbors classifier has the lowest net benefit score.
- Overall, the Decision Tree Classifier seems to perform the best on this dataset, with high accuracy, recall, and precision scores, as well as a high net benefit score.