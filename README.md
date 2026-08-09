# South Korea Fashion Retail Sales Prediction Using Random Forest

<img width="1536" height="1024" alt="ChatGPT Image Aug 9, 2026, 02_51_38 PM" src="https://github.com/user-attachments/assets/4454d80a-8737-4059-933b-396059f1eb7d" />





**Introduction**

Retail businesses generate enormous amounts of data every day. Sales transactions, customer demographics, shopping patterns, time of purchase, and location all provide valuable information that can help businesses understand and predict sales.
But turning this raw data into useful predictions is not straightforward.
In this project, I worked with a real-world South Korean commercial sales dataset to develop a machine learning model for predicting monthly sales amount.
The goal was simple:
Can machine learning learn the relationship between customer behavior, transaction patterns, commercial areas, and sales activity well enough to predict monthly retail sales?
To answer this question, I followed an end-to-end data science workflow involving data exploration, preprocessing, feature scaling, dimensionality reduction, machine learning, and model evaluation.
________________________________________
**Understanding the Dataset**

The dataset contains 85,732 records and 55 columns.
It includes information about South Korean commercial areas and different service industries. The data contains sales-related information along with customer and transaction characteristics.
Some of the major categories of information include:
•	Commercial areas

•	Service industries

•	Monthly sales amount

•	Number of sales transactions

•	Weekday and weekend sales

•	Time-of-day sales

•	Male and female customer transactions

•	Age-group transaction information

•	Other sales-related characteristics

The original dataset was in Korean, so I worked with the data while translating the relevant fields into English for easier analysis and interpretation.
________________________________________
**Defining the Machine Learning Problem**

This project is a supervised machine learning regression problem.
The target variable is:

Monthly Sales Amount

The objective is to predict a continuous sales value based on the available information about the commercial area, service industry, customer demographics, transaction activity, and sales patterns.

This makes the problem different from classification.

We are not trying to predict a category such as “high sales” or “low sales.”
Instead, we are trying to predict an actual numerical sales amount.
________________________________________
**Data Quality Analysis**
Before building a machine learning model, I first examined the quality of the dataset.
The dataset contained 85,732 observations and 55 variables.
I also checked the dataset for missing values and duplicate records.
The analysis showed:
•	No missing values

•	No duplicate records

This provided a clean starting point for the modeling process.

However, clean data does not automatically mean that the data is ready for machine learning. Feature types, scale, dimensionality, and relationships between variables still need to be considered.
________________________________________
**Exploratory Data Analysis**
The next step was exploratory data analysis.

The purpose of EDA was to understand the structure of the dataset and identify relationships between the target variable and other numerical features.

One particularly useful analysis was correlation analysis.

Several variables showed very strong relationships with monthly sales.

For example, the correlation between monthly sales and some sales-related variables was extremely high. Male sales amount had a correlation of approximately 0.984, while Friday sales amount was approximately 0.984.

Weekday sales, Thursday sales, Wednesday sales, Tuesday sales, time-period sales, and different age-group sales also showed strong relationships with the target.
This made one thing clear:

Sales are influenced by multiple dimensions of customer and transaction behavior.

Instead of looking at sales as a single number, the dataset allows us to examine sales through different perspectives such as time, demographics, and commercial activity.
________________________________________
**Preparing the Data for Machine Learning**

After understanding the data, the next stage was preparing it for the machine learning model.

Categorical information was separated from the numerical modeling features, and the target variable was separated from the predictors.

The dataset was then divided into training and testing portions.

The purpose of this split was to allow the model to learn from one portion of the data and evaluate its ability to generalize to previously unseen observations.
This is an important part of any machine learning project.

A model should not simply perform well on the data it has already seen.

The real question is:
How well does it perform on data it has never seen before?
________________________________________
**Feature Scaling**
The numerical features were scaled using RobustScaler.

This preprocessing technique was selected to make the feature values more comparable while being less sensitive to extreme observations.

This step was particularly useful because real-world commercial datasets can contain variables with very different numerical ranges.

For example, sales amounts can be extremely large compared with transaction counts.

Scaling helps create a more consistent numerical representation before applying dimensionality reduction.

________________________________________
**Reducing Dimensionality with PCA**

After scaling the features, I applied Principal Component Analysis (PCA).

PCA is a dimensionality-reduction technique that transforms the original features into a smaller number of principal components while retaining as much information as possible.

In this project, the data was reduced to two principal components.

The first component explained approximately 89.27% of the variance, while the second explained approximately 5.60%.

Together, these two components represented approximately 94.87% of the variance in the scaled feature space.

This was an interesting result.

Even though the original dataset contained many numerical variables, a large proportion of the variation could be represented using just two principal components.
________________________________________
**Building the Random Forest Model**

For the prediction task, I selected Random Forest Regression.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to produce a prediction.

It is particularly useful for tabular datasets because it can capture nonlinear relationships and interactions between variables.

In this project, the Random Forest model was trained using the PCA-transformed features.

**Model Performance**

After training the model, I evaluated it on both the training and testing datasets.

Several regression metrics were used:
•	R² Score

•	Mean Squared Error

•	Root Mean Squared Error

•	Mean Absolute Error

•	Mean Absolute Percentage Error

Using multiple metrics provides a more complete picture of model performance.
________________________________________
**Training Performance**

The Random Forest model achieved an R² score of approximately 0.9936 on the training dataset.

That means the model explained approximately 99.36% of the variation in the training target.

The training results were:

Metric	Training Result
R²	0.9936

RMSE	768,114,757

MAE	113,438,599

MAPE	16.79%

These results indicate that the model fitted the training data very strongly.
________________________________________
**Testing Performance**

The more important result is how the model performed on unseen data.

On the testing dataset, the model achieved an R² score of approximately 0.9720.

The testing results were:

Metric	Testing Result

R²	0.9720

RMSE	1,375,426,185

MAE	301,446,080

MAPE	40.02%

The test R² of 0.9720 indicates that the model captured a large proportion of the variation in monthly sales on the unseen test data.
________________________________________
**What Do These Results Tell Us?**

At first glance, the R² score of 0.9720 looks extremely strong.

And it is important.

However, looking at only one metric can give an incomplete picture.

The testing MAPE was approximately 40.02%, which means the percentage-based prediction error was considerably larger than the R² score alone might suggest.

This is an important lesson in machine learning:

A high R² score shows that the model explains the overall variation in the target, but it should be evaluated alongside error metrics to understand prediction accuracy.

That is why I evaluated the model using several different metrics instead of relying only on R².

The difference between training and testing performance is also worth observing.

The training R² was approximately 99.36%, while the testing R² was approximately 97.20%.

The decrease indicates that the model performs somewhat better on the data used for training than on unseen data, which is something that should always be examined when evaluating machine learning models.
________________________________________
**Why Random Forest?**

Random Forest was a suitable choice for this project because the dataset contains many interacting numerical variables.

Retail sales are rarely determined by one factor.

Sales can be associated with:

•	Customer demographics

•	Transaction volume

•	Time of purchase

•	Day of the week

•	Commercial area

•	Service industry

•	Other sales patterns

A tree-based ensemble model can capture complex relationships between these variables without requiring the relationships to be strictly linear.
________________________________________
**What I Learned from This Project**

This project reinforced several important lessons about practical data science.

1. Real-world data requires investigation
Before choosing a model, it is important to understand what each feature represents and how the data is structured.
2. EDA is not optional
Correlation analysis and exploratory analysis helped reveal the strong relationships between different sales dimensions and monthly sales.
3. Preprocessing matters
Scaling and dimensionality reduction were important parts of the modeling pipeline.
4. PCA can reveal hidden structure
The first two principal components captured approximately 94.87% of the variance in the scaled feature space.
5. Model evaluation needs multiple metrics
The difference between the R² score and MAPE demonstrates why a model should not be judged using only one metric.
6. Test performance matters more than training performance
A model’s ability to generalize to unseen data is one of the most important indicators of whether it can be useful beyond the training dataset.


________________________________________
**Business Perspective**

A machine learning model for sales prediction can potentially provide value to businesses by helping them understand and estimate sales patterns.
Such predictive systems could support areas such as:

•	Sales planning

•	Revenue estimation

•	Commercial-area analysis

•	Business performance monitoring

•	Customer behavior analysis

•	Resource planning

However, before deploying a model in a real business environment, additional validation would be necessary.
The model would need to be tested on future data, monitored over time, and evaluated against business requirements.
________________________________________
**Final Thoughts**

This project was more than simply training a Random Forest model.

It was an opportunity to work through a complete machine learning workflow using real-world South Korean commercial data.

Starting with 85,732 records and 55 variables, I explored the dataset, examined relationships between variables, prepared the data, applied robust scaling, reduced dimensionality using PCA, and trained a Random Forest regression model.

The final model achieved a test R² of 0.9720, showing strong predictive performance on unseen data. At the same time, the 40.02% MAPE highlighted why model evaluation should go beyond a single metric.

The biggest lesson I took from this project is simple:
Good data science is about building a strong model by understanding the data, choosing the right approach, evaluating results thoughtfully, and turning those results into meaningful insights.

This project represents one more step in my journey of applying machine learning to real-world business problems.

