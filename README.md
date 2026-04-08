# Machine Learning-Based Salary Prediction

## 📌 Project Overview

This project focuses on building a machine learning model to predict salaries based on job-related features. The goal is to analyze job postings data, identify factors influencing salary, and develop a predictive model that estimates salary accurately.

The project follows a complete machine learning pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model development, evaluation, and deployment.

---

## 📊 Dataset Description

The dataset consists of job postings data containing various attributes such as agency, job category, career level, employment type, and salary range.

Key features in the dataset include:

* Agency
* Posting Type
* Number of Positions
* Job Category
* Career Level
* Full-Time / Part-Time Indicator
* Salary Range (From & To)
* Salary Frequency
* Work Location
* Posting Date

A new feature called **Average Salary** was created using the salary range values to serve as the target variable for prediction.

---

## 🔧 Project Workflow

### 1. Data Preprocessing

* Handling missing values
* Removing irrelevant columns
* Cleaning and formatting the dataset

### 2. Feature Engineering

* Created **Average Salary** as the target variable
* Extracted **Posting Year and Posting Month** from posting date

### 3. Exploratory Data Analysis (EDA)

* Univariate analysis
* Bivariate analysis
* Correlation analysis
* Temporal trend analysis

### 4. Model Development

Multiple machine learning models were trained and evaluated:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor

### 5. Model Evaluation

Models were evaluated using the following metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

Random Forest was selected as the final model due to its superior performance.

---

## 🤖 Final Model Performance

| Model             | MAE   | RMSE  | R² Score  |
| ----------------- | ----- | ----- | --------- |
| Linear Regression | 0.244 | 0.346 | 0.976     |
| Decision Tree     | 0.059 | 0.156 | 0.995     |
| Random Forest     | 0.084 | 0.136 | **0.996** |
| XGBoost           | 0.093 | 0.139 | 0.996     |

The **Random Forest Regressor** achieved the best results with an R² score of **0.996**, indicating high prediction accuracy.

---

## 📈 Dashboard Visualization

An interactive **Power BI dashboard** was developed to visualize salary trends and insights from the dataset.

The dashboard includes:

* Salary distribution
* Salary by career level
* Salary by job category
* Salary trend over time
* Salary comparison by employment type

---

## 💻 Streamlit Application

The trained model was deployed using a **Streamlit application** that allows users to input job-related details and receive real-time salary predictions.

Prediction workflow:

1. User Input
2. Encoding of categorical variables
3. Feature alignment
4. Model prediction
5. Display predicted salary

---

## 📂 Project Structure

```
salary-prediction-ml-project/
│
├── notebooks/
│   ├── 01_data_preprocessing_and_eda.py
│   ├── 02_model_comparison.py
│   ├── 03_final_model.py
│
├── app.py
├── Salary_Prediction_Project_Report.pdf
├── README.md
```

---

## 🛠 Tools & Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib & Seaborn
* Streamlit
* Power BI

---

## 📌 Business Insights

* Career level significantly impacts salary levels
* Full-time roles offer higher salaries compared to part-time roles
* Job category plays an important role in salary variation
* Salary trends change over time depending on hiring demand

---

## 📎 Project Files

Due to file size limitations, some files are shared via Google Drive:

* Trained model files
* Power BI dashboard

Drive link: *(Add your Google Drive link here)*

---

## 👤 Author

**Sunil kumar reddy illuri**
Machine Learning & Data Science Enthusiast
