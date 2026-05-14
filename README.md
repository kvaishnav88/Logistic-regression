# Logistic-regression
Linear regression is used to predict values of quantities as a linear function of the input values. When predicting a discrete variable, such as whether a grid of pixel intensities represents 0 or 1, we need to classify the input values. Logistic regression is a simple classification algorithm for learning to make such decisions. It is a model that is used when the dependent variable is categorical. A few cases where logistic regression can be used are mentioned below:

Image segmentation and categorization
Geographic image recognition
Handwriting recognition
Determining whether a person is depressed based on the words of his social media posts
Predicting the probability of a person voting for a candidate in an election

Logistic regression falls under supervised learning; it measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic/sigmoid function. Despite the name 'logistic regression', it is not used for machine learning regression problem where the task is to predict the real-valued output. It is a classification problem that is used to predict a binary outcome (1/0, -1/1, True/False) given a set of independent variables. Logistic regression is a bit similar to linear regression, or we can say it as a generalized linear model. In linear regression, we predict a real-valued output ' y' based on a weighted sum of input variables.



                               𝑦 = 𝑐 + 𝑤1 ∗ 𝑥1 + 𝑤2 ∗ 𝑥2 ∗ 𝑤3 ∗ 𝑥3 + ...𝑤𝑛 ∗ 𝑥𝑛
Linear regression aims to estimate values for the model coefficients 𝑐
, 𝑤1
, 𝑤1
, 𝑤3
...𝑤𝑛
 and fit the training data with minimal squared error and predict the output y.

Logistic regression does the same thing, but with one addition. The logistic regression model computes a weighted sum of the input variables similar to the linear regression, but it runs the result through a special non-linear function, the logistic function or sigmoid function, to produce the output y. Here, the output is binary or in the form of 0/1 or -1/1.

The sigmoid/logistic function is given by the following equation:

𝑦=𝑙𝑜𝑔𝑖𝑠𝑡𝑖𝑐(𝑐+𝑤1∗𝑥1+𝑤2∗𝑥2∗𝑤3∗𝑥3+...𝑤𝑛∗𝑥𝑛)
 
𝑦=1/[1+𝑒−(𝑐+𝑤1∗𝑥1+𝑤2∗𝑥2∗𝑤3∗𝑥3+...𝑤𝑛∗𝑥𝑛)]
 
𝑦=1/[1+𝑒−𝑥]
 

As you can see in the below graph, it is an S-shaped curve that gets closer to 1 as the input variable's value increases above 0 and gets closer to 0 as the input variable decreases below 0. The output of the sigmoid function is 0.5 when the input variable is 0.


#Imports & Data
We will use Scikit-Learn library is to perform the logistic regression.
We will understand how to choose target and predictor variables for a logistic regression model. We have two features, prev day's return, and retruns a day prior to that. And our target variable is stored in the 'target' column.
We now define our X and y. We have two features and and our target coulmn. We split the data into train and test sets


#Logistic Regression Model
We will use the LogisticRegression function from the sklearn.linear_model library to create a logistic regression model.
Syntax:

LogisticRegression(C=1e5) 
Parameters used: 
        1. C (High value of C tells the model to give more weight to the training data than complexity penalty)

#Predict & Evaluate
We will use our trained logistic regression model to predict the next-day direction (up = 1, down/flat = 0) on the test window.

We’ll report:

Accuracy (overall hit rate)
Confusion Matrix (how many ups/downs we got right)

How to read the confusion matrix
We print it as rows = true class, columns = predicted class:

The decision-region plot
What’s on the axes?
x-axis = 1_day_lag_returns, y-axis = 2_day_lag_returns.

Shaded regions = model prediction.
The two shades show where the model predicts class 0 (down) vs 1 (up).
The straight line between them is the decision boundary where (P(\text{up}) = 0.5).

Dots = actual test days.
Each point is a day from the test set; its color indicates the true class (0 or 1).

How to interpret correctness.
A point lying inside the region of its own class is a correct prediction.
A point inside the opposite region is a mistake (false positive/negative).

Confidence intuition.
Points farther from the boundary are predicted with higher probability; points near the line are more uncertain.

What the slope means.
The tilt of the boundary shows how the two lagged returns trade off: e.g., a higher 1-day lag can offset a negative 2-day lag to still predict “up”.


The decision-region plot
What’s on the axes?
x-axis = 1_day_lag_returns, y-axis = 2_day_lag_returns.

Shaded regions = model prediction.
The two shades show where the model predicts class 0 (down) vs 1 (up).
The straight line between them is the decision boundary where (P(\text{up}) = 0.5).

Dots = actual test days.
Each point is a day from the test set; its color indicates the true class (0 or 1).

How to interpret correctness.
A point lying inside the region of its own class is a correct prediction.
A point inside the opposite region is a mistake (false positive/negative).

Confidence intuition.
Points farther from the boundary are predicted with higher probability; points near the line are more uncertain.

What the slope means.
The tilt of the boundary shows how the two lagged returns trade off: e.g., a higher 1-day lag can offset a negative 2-day lag to still predict “up”.


Key takeaways
What we built: A simple logistic regression that predicts next-day direction (0 = down, 1 = up) using lagged returns.
No look-ahead: We used prior-day lags as features and kept the time order; the last 20% of data was our test set.
Outputs: predict() gives class labels; predict_proba()[:, 1] gives P(up) for each day.
Evaluation: Accuracy and the confusion matrix show overall hits and where the model confuses ups vs downs.
Decision boundary: With two features, the model draws a straight line separating predicted up vs down regions.
Limitations: Accuracy can be inflated by class imbalance; treat results as a teaching demo, not trading advice.
























