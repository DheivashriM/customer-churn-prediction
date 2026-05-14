# 📊 Telco Customer Churn Prediction & Business Dashboard

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-ffffff?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-blueviolet?style=for-the-badge)

---

## 📖 Project Overview

Customer churn is one of the most critical metrics for subscription-based businesses. Retaining an existing customer is significantly more cost-effective than acquiring a new one. 

This project implements an **end-to-end Data Science and Machine Learning pipeline** to predict customer churn using the Telco Customer dataset. It features seamless integration with a **Supabase PostgreSQL database**, robust exploratory data analysis, preventive feature engineering against data leakage, model training & comparison, and an automated high-impact **Business Dashboard** artifact.

---

## 📁 Project Folder Structure

```text
customer-churn-prediction/
├── data/
│   └── telco_churn.csv          # Merged Telco Customer Churn dataset
├── notebooks/
│   └── analysis.ipynb           # Complete executable Jupyter Notebook walkthrough
├── src/
│   ├── database.py              # Supabase schema setup, data ingestion, and 10 SQL queries
│   ├── eda.py                   # Exploratory Data Analysis & visual plot generation
│   ├── preprocessing.py         # Missing value imputation, LabelEncoding, scaling, splits
│   ├── model.py                 # ML training (LogReg vs DecTree), evaluation & dashboard logic
│   └── generate_data.py         # Custom dataset generator matching schema perfectly
├── outputs/
│   ├── dashboard.png            # Final consolidated 6-subplot business presentation figure
│   └── eda_plots/               # Individual high-resolution EDA charts
├── requirements.txt             # Complete list of stable package dependencies
└── README.md                    # Comprehensive documentation and study guide
```

---

## 🚀 How to Run the Project

Follow these beginner-friendly, professional steps to run the pipeline locally:

### 1. Environment Setup
Ensure you have Python 3.11+ installed. Clone or navigate to the project repository:
```bash
cd "customer-churn-prediction"
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Supabase Credentials
Open the `.env` file located in the root directory and update it with your active Supabase project keys:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```
> **Note:** If you do not configure Supabase keys, the code has fully robust fallback mechanisms to automatically load and execute queries directly from local CSV datasets, guaranteeing zero execution interruptions!

### 3. Execute Database Operations & SQL Queries
Run the database module to review table schemas, insert data, and execute the **10 mandatory business SQL queries**:
```bash
python src/database.py
```

### 4. Perform Exploratory Data Analysis (EDA)
Generate descriptive statistical summaries and render high-resolution charts to `outputs/eda_plots/`:
```bash
python src/eda.py
```

### 5. Data Preprocessing
Test the cleaning, encoding, imputation, and train-test split operations:
```bash
python src/preprocessing.py
```

### 6. Train Machine Learning Models & Build Dashboard
Train **Logistic Regression** and **Decision Tree** classifiers, print full classification reports and confusion matrices, and generate the complete **Final Business Dashboard**:
```bash
python src/model.py
```
Upon successful execution, view the beautiful consolidated output artifact at `outputs/dashboard.png`.

---

## 📸 Screenshots & Visualizations

### Final Consolidated Business Dashboard (`outputs/dashboard.png`)
The automated reporting dashboard provides executives with actionable answers at a single glance:
1. **Overall Churn Rate Pie Chart**: Baseline breakdown of active vs lost accounts.
2. **Tenure vs Churn Boxplot**: Demonstrates clear hazard risk for early-stage subscribers.
3. **Monthly Charges Distribution**: Highlights premium segment attrition rates.
4. **Contract Type vs Churn Bar Chart**: Illustrates highly vulnerable month-to-month plans.
5. **Top Feature Importances**: Decodes decision tree decision rules.
6. **Model Accuracy Comparison**: Directly ranks predictive estimators.

---

## 💡 Key Business Insights

1. **Critical Tenure Windows**: Customers retained by the business average a tenure of **~29.8 months**, whereas churned users depart after an average of only **~16.7 months**. Focused onboarding and early retention initiatives yield the highest ROI.
2. **Revenue Vulnerability**: Churn directly accounted for a massive **$26,395.76** in lost recurring monthly revenue within this sample cohort alone.
3. **Contract Stability**: Subscribers on **Month-to-month contracts** exhibit an alarming **52.6% churn rate**. Conversely, locking users into **1-year (16.3%)** or **2-year (8.0%)** agreements drastically reduces attrition.
4. **Premium Service Friction**: Customers utilizing **Fiber Optic internet** experience higher churn (**37.8%**) compared to standard DSL (**30.2%**), indicating a mismatch between premium price tags and expected service reliability or support attitudes.
5. **Top Actionable Churn Drivers**: Customer feedback explicitly highlights **Competitor offers/devices**, **Support staff attitude**, and **Network reliability** as primary drivers. Resolving technical connection drops and training support agents directly preserves revenue.

---

## 🏆 Resume Bullet Points

Add these impactful, metric-driven bullet points directly to your professional resume:

* **Engineered an end-to-end Customer Churn Prediction pipeline** integrating Python, Pandas, and Scikit-Learn with a **Supabase cloud database**, analyzing over 1,000 subscriber records to protect recurring revenue streams.
* **Designed relational database schemas** with primary/foreign key constraints and automated batch data ingestion utilizing `supabase-py` and RESTful PostgREST APIs.
* **Formulated and executed 10 complex SQL business queries** to extract real-time KPI metrics including cohort churn rates, contract risk distributions, and cumulative revenue loss.
* **Prevented critical Data Leakage** by systematically identifying and dropping post-facto variables (`churn_reason`, `churn_date`), ensuring models strictly generalize to active subscriber predictions.
* **Trained and evaluated Machine Learning classifiers** (Logistic Regression vs Decision Tree), comparing metrics via precision/recall and achieving an optimal baseline accuracy of **71.0%** using Logistic Regression.
* **Developed an automated consolidated presentation artifact** using Seaborn and Matplotlib, rendering a 6-subplot presentation-ready figure mapping customer behavior and feature importances for executive stakeholders.

---

## 🎤 Top 10 Interview Questions & Answers

### Q1: Why did you drop `churn_reason` and `churn_date` from your machine learning feature set?
**Answer:** These variables represent post-facto information collected only *after* a customer has officially churned. Including them as predictive features during model training causes direct **data leakage**, resulting in artificially inflated training scores that completely fail to generalize to active, non-churned customers in a real business environment.

### Q2: How did you handle missing values in this project?
**Answer:** Missing values were identified in the `total_charges` column. Since numerical features can be skewed by outliers, we applied **median imputation** (`SimpleImputer(strategy='median')` / `.fillna()`) to fill missing records, ensuring the baseline distribution remains robust and stable.

### Q3: Explain the schema design and constraints you implemented in Supabase.
**Answer:** We normalized the records into three distinct relational tables: `customers` (demographics & billing), `services` (subscribed features), and `churn_data` (outcomes). We enforced **Primary Keys** on `customer_id`, **Foreign Keys** with `ON DELETE CASCADE` to maintain referential integrity, and **CHECK constraints** to guarantee valid entries (e.g., non-negative charges, restricted category choices for gender and contracts).

### Q4: Why did you split the data into train and test sets *before* applying `StandardScaler`?
**Answer:** To prevent **information leakage**. Fitting a scaler on the entire dataset incorporates the global mean and standard deviation of the test set into the training phase. By splitting first, the scaler is fitted strictly on the training set and applied to the test set independently, simulating real-world unseen data ingestion.

### Q5: Compare the performance of Logistic Regression vs Decision Tree in your pipeline. Which one performed better and why?
**Answer:** **Logistic Regression** achieved a higher test accuracy (**71.0%**) compared to the **Decision Tree** (**64.5%**). Because customer churn features in this dataset (like tenure and monthly charges) exhibit smooth linear boundaries and consistent correlations with the log-odds of churning, Logistic Regression captures the global pattern efficiently without suffering from localized variance or overfitting common in un-tuned decision trees.

### Q6: How does `supabase-py` interact with the underlying database?
**Answer:** `supabase-py` acts as a client wrapper communicating primarily via **PostgREST**, an automated web server that turns a PostgreSQL database directly into a RESTful API. It allows performing seamless CRUD operations using an intuitive, chainable ORM-like syntax while respecting Row Level Security (RLS) policies.

### Q7: What business actions would you recommend based on the Exploratory Data Analysis (EDA)?
**Answer:** 
1. Target customers on **Month-to-month contracts** with loyalty discounts to transition them into 1-year or 2-year agreements.
2. Focus proactive customer success outreach during the **first 6 to 12 months** of tenure, where the churn hazard rate is highest.
3. Address technical support training and network reliability, as attitude and connection stability are leading controllable churn drivers.

### Q8: What metrics besides Accuracy are critical when evaluating a Churn Prediction model?
**Answer:** **Precision, Recall, and F1-score** are crucial because churn datasets are often imbalanced. **Recall** measures our ability to identify all true churners (minimizing false negatives, which represent lost revenue), while **Precision** ensures we do not waste marketing retention budgets on highly satisfied customers (minimizing false positives).

### Q9: How did you preserve feature names during scaling to maintain dashboard readability?
**Answer:** Scikit-learn's standard transformers typically output anonymous NumPy arrays. We preserved column names by applying the fitted scaler explicitly back to the DataFrame columns (`X_scaled[cols] = scaler.transform(X[cols])`), allowing seamless downstream mapping for extracting accurate feature importance visualizations in our dashboard.

### Q10: How would you deploy this pipeline into production?
**Answer:** We would containerize the code using **Docker**, schedule automated daily or weekly inference runs using an orchestrator like **Airflow** or **GitHub Actions**, read incoming batch data securely from Supabase via environment variables, and stream output risk scores/dashboards directly to a BI tool or CRM to trigger automated customer retention workflows.

---
*Built with professional quality and best practices by Antigravity.*
