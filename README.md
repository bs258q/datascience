# Data Drift Analysis for Marketing Campaigns

This repository contains code and resources for performing data drift analysis on a marketing campaign dataset. Data drift is a common problem in machine learning where the distribution of the input data changes over time, leading to a decrease in model performance. This project provides tools and techniques to detect and understand data drift, enabling proactive model maintenance and improved prediction accuracy.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Dependencies](#dependencies)
4.  [Code Overview](#code-overview)
5.  [Data Preprocessing](#data-preprocessing)
6.  [Data Drift Detection Methods](#data-drift-detection-methods)
    *   [Kolmogorov-Smirnov Test (Numerical Features)](#kolmogorov-smirnov-test-numerical-features)
    *   [Chi-squared Test (Categorical Features)](#chi-squared-test-categorical-features)
    *   [Evidently Library](#evidently-library)
7.  [Running the Analysis](#running-the-analysis)
8.  [Interpreting the Results](#interpreting-the-results)
    *   [P-value](#p-value)
    *   [Feature Distributions](#feature-distributions)
    *   [Evidently Reports](#evidently-reports)
9.  [Addressing Data Drift](#addressing-data-drift)
10. [Contributing](#contributing)
11. [License](#license)

## 1. Introduction

This project addresses the challenge of data drift in marketing campaign analysis. By implementing data drift detection techniques, we can:

*   Identify when the characteristics of our customer data change.
*   Understand which features are most affected by drift.
*   Make informed decisions about retraining or updating our models.

## 2. Dataset

The project uses a dataset named `marketing_campaign.csv`. This dataset (or one similar to it - see below) contains information about customers and their responses to marketing campaigns. The key columns include:

*   **Response:** (Target Variable) Indicates whether a customer responded positively to a marketing campaign (e.g., made a purchase). Typically binary (0 or 1).
*   **Customer Attributes:** Demographic information, spending habits, engagement metrics.

**Example Dataset Structure (based on the example you provided):**
content_copy
download
Use code with caution.
Markdown

ID,Year_Birth,Education,Marital_Status,Income,Kidhome,Teenhome,Dt_Customer,Recency,MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Complain,Z_CostContact,Z_Revenue,Response
5524,1957,Graduation,Single,58138,0,0,04-09-2012,58,635,88,546,172,88,88,3,8,10,4,7,0,0,0,0,0,0,3,11,1
2174,1954,Graduation,Single,46344,1,1,08-03-2014,38,11,1,6,2,1,6,2,1,1,2,5,0,0,0,0,0,0,3,11,0
...

**Note:** A sample `marketing_campaign.csv` file is not included directly in this repository for data privacy reasons. The code is designed to work with a similar dataset containing customer attributes and a response variable. You'll need to provide your own dataset in the same format.

## 3. Dependencies

To run the data drift analysis, you will need the following Python libraries:

*   **pandas:** For data manipulation and analysis.
*   **scikit-learn:** For data preprocessing (e.g., label encoding).
*   **category_encoders:** For target encoding of categorical features (optional, but recommended).
*   **scipy:** For statistical tests (Kolmogorov-Smirnov, Chi-squared).
*   **evidently:** A Python library for data and model evaluation and drift detection.

You can install these libraries using pip:

```bash
pip install pandas scikit-learn category_encoders scipy evidently
content_copy
download
Use code with caution.
4. Code Overview

The core code for the data drift analysis is in data_drift_analysis.py (or a similar name - adjust to your specific file). It performs the following steps:

Data Loading and Preprocessing: Loads the data, handles missing values, and encodes categorical features.

Data Splitting: Splits the data into reference (training) and current (testing) sets.

Data Drift Detection:

Applies the Kolmogorov-Smirnov test to numerical features.

Applies the Chi-squared test to categorical features.

Creates and runs Evidently reports for data drift and target drift.

Result Visualization: Generates HTML reports using Evidently to visualize the data drift results.

5. Data Preprocessing

The code includes basic data preprocessing steps to handle missing values and encode categorical features. You may need to adapt these steps based on the characteristics of your specific dataset.

Missing Value Handling: Missing values are replaced with the mean of the respective column. This is a simple approach. Consider using more sophisticated methods like imputation with median, mode, or using more advanced imputation techniques from scikit-learn.

Categorical Encoding: Categorical features are encoded using label encoding. This converts categorical values into numerical values. For higher-cardinality categorical features, consider using target encoding or other encoding methods from the category_encoders library. One-hot encoding should also be considered.

6. Data Drift Detection Methods

The project uses the following methods to detect data drift:

Kolmogorov-Smirnov Test (Numerical Features)

The Kolmogorov-Smirnov (KS) test is used to determine if two independent samples are drawn from the same continuous distribution. A low p-value (typically < 0.05) suggests data drift.

Chi-squared Test (Categorical Features)

The Chi-squared test is used to assess the independence of two categorical variables. In the context of data drift, we use it to check if the distribution of a single categorical variable has changed between two datasets. A low p-value (typically < 0.05) suggests data drift. The implementation handles the case where the data does not contain all of the same values.

Evidently Library

The Evidently library provides a comprehensive and user-friendly way to detect data drift. It automates the data drift detection process, provides visualizations, and generates insightful reports. The code uses two key components from Evidently:

DataDriftPreset: A collection of metrics specifically designed to detect data drift across all features.

TargetDriftPreset: Metrics for evaluating the performance of the regression model and detecting changes in model performance over time. This specifically looks for drift in the target variable.

7. Running the Analysis

Prepare Your Data: Ensure your dataset is named marketing_campaign.csv (or update the code to reflect your filename) and placed in the same directory as the script.

Run the Python Script: Execute the data_drift_analysis.py (or similar) script using Python:

python data_drift_analysis.py
content_copy
download
Use code with caution.
Bash

This will generate HTML reports (data_drift_report.html and target_drift_report.html) in the same directory.

8. Interpreting the Results

After running the analysis, you'll need to interpret the results to identify data drift and its potential impact on your models.

P-value

The statistical tests (KS and Chi-squared) provide a p-value for each feature. The p-value represents the probability of observing the data if there is no data drift.

Low P-value (typically < 0.05 or 0.01): Indicates statistically significant evidence of data drift. Reject the null hypothesis (that the distributions are the same).

High P-value (typically >= 0.05 or 0.01): Indicates that there is not enough evidence to conclude that the data has drifted. Fail to reject the null hypothesis.

Feature Distributions

The Evidently reports visualize the distributions of each feature in the reference and current datasets. Examine these distributions to understand the nature of the data drift. Are the distributions shifting, becoming more spread out, or changing in other ways?

Evidently Reports

The Evidently HTML reports provide a comprehensive overview of the data drift analysis. Look for the following key information:

Summary Metrics: The reports summarize the overall data drift, highlighting the features with the most significant drift.

Drift Scores: Evidently calculates drift scores for each feature. These scores provide a quantitative measure of the degree of data drift.

Visualizations: The reports include interactive visualizations that help you understand the nature and extent of data drift.

9. Addressing Data Drift

If you detect significant data drift, you may need to take action to mitigate its impact on your models. Some common strategies include:

Retraining Your Model: The most common approach is to retrain your model using the most recent data.

Feature Engineering: Re-evaluate your feature engineering process. You may need to create new features or modify existing ones to better capture the current data distribution.

Data Monitoring: Continuously monitor your data for drift to ensure that your models remain accurate.

Adaptive Models: Consider using models that are more robust to data drift or that can adapt to changing data distributions (e.g., online learning algorithms).

10. Contributing

Contributions to this project are welcome! If you have ideas for improvements, bug fixes, or new features, please submit a pull request. Please follow these guidelines:

Use clear and concise commit messages.

Write unit tests for new code.

Follow the project's coding style.

11. License

This project is licensed under the MIT License.

**Important Notes:**

*   **`LICENSE` File:** Create a file named `LICENSE` (all caps) in your repository and include the full text of the MIT License.
*   **Adapt to Your Files:**  Modify the README to accurately reflect the names of your Python scripts and data files.
*   **Real Data Processing:** The sample code for data preprocessing is very basic. In a real-world project, you'll need to implement more robust methods for handling missing values, encoding categorical features, and scaling numerical features.
*   **Evidently Customization:**  Evidently is highly customizable. Explore the library's documentation to learn how to create custom metrics, tests, and reports.
*   **Detailed Output:** In your own projects, make sure to write informative print statements so it's very clear what is being analyzed and what the results are.
*   **Adjust to your Specific License:** I've suggested the MIT License, but you are free to use any open-source license you choose.
*   **Add to Git:** After adding the README and LICENSE, make sure to commit these and push to Github.

This provides a good starting point for a well-documented GitHub repository! Remember to personalize this README further to reflect the specific details of your project.
content_copy
download
Use code with caution.
