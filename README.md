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
