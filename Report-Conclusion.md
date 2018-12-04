---
title: Results and Conclusion
notebook: Report-ConclusionMD.ipynb
nav_include: 5
---

## Contents
{:.no_toc}
*  
{: toc}


## 1. Summary

By using 2 thigh IMUs and 1 torso IMU, we could achieve slope prediction accuracy up to **99.9%** using indiviaully trained models for 3 class classification. Depending on the feasibiliity of implementing individually trained models, we also demontrated that a universal 3-class slope classification model can achieve accuracy up to **90.5%**

To achieve higher prediction resolution, we also formulated the slope estimation problem as a 7-class classification and a regression problem. The regression model outperforms the 7-classification model in the universal model (one model for all subject) case. The regression model prediction accuracy is **75.4%**, while the 7-class classification accuracy is only *54.2%*. One explanation for the superior performance of the regression model is the fact that slope gradient is an ordinal and continuous variable, and a linear regression model would taken advantage of this property.
In the subject-specific models, classification using random forest has a very high accuracy of **99.2%**. Training individual models for each subject shows promise for high accurarcy slope classification. The regression model showed worse performance in the subject-specific models with the highest test accuracy achieved being *93%*.

In summary, for the exosuit slope estimation problem using only torso and thigh IMUs, the best classification model is **random forest**, and the best regression method is **linear regression with 90% explained PCA components**.

## 2. Results

### 1) 3-Class Classification

We tried various modeling methods for the 3-class slope gradient estimation problem. We chose to use a decision tree with unlimited depth as a baseline model. The train score for the baseline model is 1.0, while the test score is 0.817. This is expected as an unlimted depth tree will overfit to the training data.

We used two different cross validation methods to determine the best tree depths and other hyperparameters. The first is normal CV using *GridSearchCV*, which spilts the train and validation sets with no particular restrictions. The second is Leave-One-Subject-Out CV (LOSO CV), which uses k-fold cross validation with each fold being data from one subject. We feel that LOSO CV is important because the goal is the train the model so that it can perform well on unseen data from a new subject.

In addition to trees, we also tried Logistic Regression Classifier and Support Vector Classifier. We used a RBF kernel (nonlinear) on the Support Vector Classifier as inspired from the literature review.

Moreover, we implemented a Gaussian Mixture Model (GMM) as it has been successfully used by Professor Nigel Lovell's reserach group. However, we found that the GMM model performed poorly on our data potentially due to the nature of unsupervised learning or the inaccurate assumption of a Gaussian data distribution. Also, we should point out that the IMU data we are using is a much more limited set that what was used in Lovell's publication. We did not have shank and foot IMUs whereas they had used those IMU outputs as additional predictors.

A summary of all models and findings are shown below.

**Table 1. Summary of All Methods and Findings**
![Table1](/Images/Summary_Table_1.png)

Comparing our baseline Tree model to our optimal random forest model, we see that the performance has improved greatly. Graphically, we see that there are fewer misclassification in the random forest model.

**Fig 1. Baseline vs. Best Model**
![Table1](/Images/con_graph1.png)

From the confusion matrix of our best random forest model, we see that downhill cases have the highest classification accuracy. Errors when predicting flat ground cases are more likely to be downhill than uphill, uphill cases can be wrongly predicted as either flat ground of downhill.

**Fig 2. Confusion Matrix for Random Forest Model**

![Table1](/Images/con_cm1.png)

### - Individually Trained Classification

For the subject-specific classification, we applied the best model from the universal model described above. Building random forest models for each of the 6 subjects, we see that the classification accuracy is near perfect.

**Table 2. Summary of All Methods and Findings for Individually Trained 3-class Problem**
![Table1](/Images/Summary_table_2.png)

**Fig 3. Subject-specific Random Forest Model**
![Table1](/Images/con_graph2.png)

### 2) 7-Class Classification

For the 7-class classification problem, we used a variety of methods. The baseline model of decision tree with unlimited depth showed overfitting on the traning data, and performed poorly on the test set. From there, we modified the baseline model using the two types of cross validation methodds mentioned above. The best model was random forest with maximum depth of 43 and 16 trees. A summary of prediction accuracies of various models is shown belown.

**Table 3. Summary of All Methods and Findings for 7-class Classification**
![Table1](/Images/Summary_table_3.png)

Visually comparing the prediction accuracy of our baseline tree model to our best random forest model, we see great improvements have been made. It seems like there are fewer misclassification cases of classifying the slope as lower than the true value.

**Fig 4. Baseline vs. Best Model**
![Table1](/Images/con_graph3.png)

From the confusion matrix of the best random forest model, there is a more likely chance of wrongly classifying a slope as lower than the true value as opposed to higher. This observation is also shown in the previous figure. Also, for all classes, we have significant number of true positive predictions.

**Fig 5. Baseline vs. Best Model**
![Table1](/Images/con_cm2.png)

### - Individually Trained Classification

We used the best model, random forest, for training individual classification models.
From the results, we see that the average train score can reach 1.0, and the average test scorey is also very high, 0.992. The prediction accuracy is greatly improved by using subject-specific models.

**Table 4. Summary of Findings for Individually Trained 7-class Classification**
![Table1](/Images/Summary_table_4.png)

The prediction accuracies of this individual model is shown to be almost perfect. The results are graphed below.

**Fig 6. Subject-specific Random Forest Model**
![Table1](/Images/con_graph4.png)

### 3) Linear Regression

For the last part of this slope prediction problem, we used various regression models to test the prediction performance. The baseline model used is a simple linear regression model with all predictors, and the accuracy is around 0.73. The converted test accuracy is measured from rounding the predicted slope to the nearest discrete slope we collected data at (-15, -10, .. 0, .. 20). 

From the baseline model,  we applied Lasso and Ridge regularization methods and reduce the model complexity and avoid overfitting. The performance did improve, and we see that Lasso regression eliminated more predictors than Ridge regression as expected. Lasso regression helped to perform feature selection in addition to shrinking regression coefficients.

The best performing model we was is linear ression with PCA components. We chose to use 61 componenets as it explained 90% of the variance in predictors. This model is also computationally light as it uses fewer predictors, but it is more difficult to interpret and relate to human biomechnics due the nature of PCA components.  

**Table 5. Summary of Findings for Regression Models**
![Table1](/Images/Summary_table_5.png)

We compared the performance of the baseline model and the best regression model, and great improvements are shown. In addition, the best performing regression model as avoided making outlier predictions.

**Fig 7. Baseline vs. Best Model (Regression)**
![Table1](/Images/con_graph5.png)

### - Individually Trained Model

Applying the best regression model found above to the individually trained modeling approach, we found even higher prediction accuracies. Note although the converted test accuracy (from rounding predictions) is quite high (0.93), it is lower than that of the 7-class classification random forest model. This is a very interesting finding as the regression model out performed the 7-class classifier when only one universal model is used.

**Table 6. Summary of Findings for Individually Trained Regression Model**
![Table1](/Images/Summary_table_6.png)

The regression results for each subject is shown below. 

**Fig 8. Subject-specific Regression Models**
![Table1](/Images/con_graph6.png)

## 3. Conclusion & Future Work

In this work, we showed the possibility of estimating walking slope using only thigh and torso inertial motion sensors. All prediction accuracies from top performing models for each problem formulation are above 90%, with the exception of 7-class classification using one universal model. There were interesting findings comparing the performance of a classifier to the performance of a regressor, as one method outperforms the other depending on when the training data is subject specific. With this in mind, we recommend using a classifier if training individual models for each subject is feasible for exosuit development and testing; if not, then a regression model is preferred.

There are several challenges related to translating these models to actual exosuits that we can consider as next steps. In order to use the current models on an actual exosuit, we need to account for calculation time on embedded systems. We need to greatly simplify the prediction algorithm complexity in order for the embedded system implementation to not interfere with other time-dependent algorithms. In addition, we need to investigate whether using individually trained prediction models is feasible for exosuit research and translation. 

For other future directions, we recognize that our current models lack interpretability, which can poentially be resolved by engineering new features based on biomechanics knowledge. The current appoach is purely data driven, thus including any biomechanics insight could greatly improve interpretability as well as reduce algorithm complexity.





Introduction
============

Here is the text of your introduction.

$$\label{simple} \alpha = \sqrt{ \beta }$$

Subsection Heading Here
-----------------------

Write your subsection text here.

Conclusion
==========

Write your conclusion here.
