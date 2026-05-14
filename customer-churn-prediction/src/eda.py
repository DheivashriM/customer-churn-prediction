import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns

# Ensure src directory is in path for robust importing
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from database import fetch_all_data

def run_eda(save_plots=True):
    """
    Performs Exploratory Data Analysis (EDA) on the merged dataset.
    Outputs summary statistics, datatypes, and null value counts.
    Generates required visualizations:
    - Churn distribution (pie chart)
    - Tenure distribution (histogram)
    - Monthly charges vs churn (boxplot)
    - Correlation heatmap
    - Churn rate by contract type (bar chart)
    """
    df = fetch_all_data()
    if df.empty:
        print("[Error] No data available for EDA.")
        return
        
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("="*60)
    
    # 1. Shape check
    print(f"\n1. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Datatypes check
    print("\n2. Column Datatypes:")
    print(df.dtypes.to_string())
    
    # 3. Null values check
    print("\n3. Null Values Check:")
    null_counts = df.isnull().sum()
    print(null_counts.to_string())
    
    # Set global plotting style for professional quality
    plt_sns.set_theme(style="whitegrid")
    output_dir = os.path.join(base_dir, "../outputs/eda_plots")
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        
    # --- Plot 1: Distribution of churned vs not churned (pie chart) ---
    plt.figure(figsize=(6, 6))
    churn_counts = df['churned'].value_counts()
    plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=['#4C72B0', '#C44E52'], explode=[0, 0.08],
            shadow=True, textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title('Distribution of Churned vs Retained Customers', fontsize=14, weight='bold', pad=15)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "churn_distribution.png"), dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: {os.path.join(output_dir, 'churn_distribution.png')}")
    plt.close()
    
    # --- Plot 2: Tenure distribution (histogram) ---
    plt.figure(figsize=(8, 5))
    plt_sns.histplot(data=df, x='tenure', bins=24, kde=True, color='#55A868')
    plt.title('Customer Tenure Distribution (Months)', fontsize=14, weight='bold', pad=15)
    plt.xlabel('Tenure (Months)', fontsize=12)
    plt.ylabel('Count of Customers', fontsize=12)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "tenure_distribution.png"), dpi=300, bbox_inches='tight')
        print(f"Saved plot: {os.path.join(output_dir, 'tenure_distribution.png')}")
    plt.close()
    
    # --- Plot 3: Monthly charges vs churn (boxplot) ---
    plt.figure(figsize=(8, 5))
    plt_sns.boxplot(data=df, x='churned', y='monthly_charges', palette=['#4C72B0', '#C44E52'])
    plt.title('Monthly Charges vs Churn Status', fontsize=14, weight='bold', pad=15)
    plt.xlabel('Churned', fontsize=12)
    plt.ylabel('Monthly Charges ($)', fontsize=12)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "monthly_charges_vs_churn.png"), dpi=300, bbox_inches='tight')
        print(f"Saved plot: {os.path.join(output_dir, 'monthly_charges_vs_churn.png')}")
    plt.close()
    
    # --- Plot 4: Correlation heatmap ---
    # Convert numerical columns for correlation matrix
    # Convert binary/categorical features to numeric approximations for rich visualization
    corr_df = df[['age', 'tenure', 'monthly_charges']].copy()
    # Convert total charges safely to float
    corr_df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    corr_df['churn_numeric'] = df['churned'].map({'Yes': 1, 'No': 0})
    
    plt.figure(figsize=(8, 6))
    corr_matrix = corr_df.corr()
    plt_sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap', fontsize=14, weight='bold', pad=15)
    if save_plots:
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
        print(f"Saved plot: {os.path.join(output_dir, 'correlation_heatmap.png')}")
    plt.close()
    
    # --- Plot 5: Churn rate by contract type (bar chart) ---
    plt.figure(figsize=(8, 5))
    contract_churn = df.groupby('contract_type')['churned'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    contract_churn.columns = ['contract_type', 'churn_rate_%']
    
    ax = plt_sns.barplot(data=contract_churn, x='contract_type', y='churn_rate_%', palette='viridis')
    plt.title('Churn Rate by Contract Type', fontsize=14, weight='bold', pad=15)
    plt.xlabel('Contract Type', fontsize=12)
    plt.ylabel('Churn Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add data labels on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 8), 
                    textcoords='offset points', weight='bold')
                    
    if save_plots:
        plt.savefig(os.path.join(output_dir, "churn_by_contract.png"), dpi=300, bbox_inches='tight')
        print(f"Saved plot: {os.path.join(output_dir, 'churn_by_contract.png')}")
    plt.close()
    print("\nEDA Completed successfully!")
    print("="*60)

if __name__ == "__main__":
    run_eda()
