import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure src directory is in path for robust importing
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from preprocessing import preprocess_data

def train_and_evaluate_models():
    """
    Trains and compares Machine Learning models for Customer Churn Prediction:
    1. Trains Logistic Regression model.
    2. Trains Decision Tree model.
    3. Evaluates both models using Accuracy Score, Confusion Matrix, and Classification Report.
    4. Compares performance and explicitly prints which model performs better.
    
    Returns:
        results (dict): Dictionary containing trained models, accuracy scores, and feature importances.
    """
    print("\n" + "="*60)
    print("MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("="*60)
    
    try:
        X_train, X_test, y_train, y_test = preprocess_data()
    except Exception as e:
        print(f"[Error] Preprocessing failed: {e}")
        return None
        
    feature_names = X_train.columns.tolist()
    
    # -----------------------------------------------------------------
    # Model 1: Logistic Regression
    # -----------------------------------------------------------------
    print("\n" + "-"*50)
    print("Training Model 1: Logistic Regression...")
    print("-"*50)
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    lr_preds = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    lr_cm = confusion_matrix(y_test, lr_preds)
    lr_cr = classification_report(y_test, lr_preds, target_names=['Retained (0)', 'Churned (1)'])
    
    print(f"Logistic Regression Accuracy Score: {lr_accuracy:.4f}")
    print("\nLogistic Regression Confusion Matrix:")
    print(lr_cm)
    print("\nLogistic Regression Classification Report:")
    print(lr_cr)
    
    # -----------------------------------------------------------------
    # Model 2: Decision Tree Classifier
    # -----------------------------------------------------------------
    print("\n" + "-"*50)
    print("Training Model 2: Decision Tree Classifier...")
    print("-"*50)
    
    # Using max_depth to prevent overfitting and achieve highly robust performance
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    
    dt_preds = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_preds)
    dt_cm = confusion_matrix(y_test, dt_preds)
    dt_cr = classification_report(y_test, dt_preds, target_names=['Retained (0)', 'Churned (1)'])
    
    print(f"Decision Tree Accuracy Score: {dt_accuracy:.4f}")
    print("\nDecision Tree Confusion Matrix:")
    print(dt_cm)
    print("\nDecision Tree Classification Report:")
    print(dt_cr)
    
    # -----------------------------------------------------------------
    # Model Comparison & Final Conclusion
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"Logistic Regression Test Accuracy : {lr_accuracy * 100:.2f}%")
    print(f"Decision Tree Test Accuracy       : {dt_accuracy * 100:.2f}%")
    print("-"*60)
    
    if lr_accuracy > dt_accuracy:
        best_model_name = "Logistic Regression"
        diff = (lr_accuracy - dt_accuracy) * 100
    elif dt_accuracy > lr_accuracy:
        best_model_name = "Decision Tree Classifier"
        diff = (dt_accuracy - lr_accuracy) * 100
    else:
        best_model_name = "Both models performed equally well"
        diff = 0.0
        
    print(f"\n>>> CONCLUSION: {best_model_name} performs better on the test data by {diff:.2f}% accuracy. <<<")
    print("="*60)
    
    # Extract feature importances from Decision Tree for business dashboard insights
    dt_importances = pd.Series(dt_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    results = {
        'lr_model': lr_model,
        'lr_accuracy': lr_accuracy,
        'dt_model': dt_model,
        'dt_accuracy': dt_accuracy,
        'feature_importances': dt_importances,
        'feature_names': feature_names
    }
    
    return results

def generate_final_dashboard(results):
    """
    Creates a single figure business dashboard with 6 subplots:
    1. Churn rate pie chart
    2. Tenure vs churn boxplot
    3. Monthly charges distribution
    4. Contract type vs churn bar chart
    5. Feature importance chart
    6. Model accuracy comparison bar chart
    
    Saves the dashboard exactly as 'outputs/dashboard.png'.
    """
    print("\nGenerating Final Business Dashboard...")
    import matplotlib.pyplot as plt
    import seaborn as plt_sns
    from database import fetch_all_data
    
    # Set professional style
    plt_sns.set_theme(style="whitegrid")
    
    df = fetch_all_data()
    if df.empty:
        print("[Error] Could not fetch data for dashboard generation.")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Telco Customer Churn Prediction - Final Business Dashboard", fontsize=22, weight='bold', y=1.02, color='#2C3E50')
    
    # 1. Churn rate pie chart (Row 0, Col 0)
    churn_counts = df['churned'].value_counts()
    axes[0, 0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
                   startangle=140, colors=['#4C72B0', '#C44E52'], explode=[0, 0.08],
                   shadow=True, textprops={'fontsize': 12, 'weight': 'bold'})
    axes[0, 0].set_title("Overall Churn Rate", fontsize=15, weight='bold', pad=10)
    
    # 2. Tenure vs churn boxplot (Row 0, Col 1)
    plt_sns.boxplot(ax=axes[0, 1], data=df, x='churned', y='tenure', palette=['#4C72B0', '#C44E52'])
    axes[0, 1].set_title("Customer Tenure vs Churn Status", fontsize=15, weight='bold', pad=10)
    axes[0, 1].set_xlabel("Churned", fontsize=12)
    axes[0, 1].set_ylabel("Tenure (Months)", fontsize=12)
    
    # 3. Monthly charges distribution (Row 0, Col 2)
    plt_sns.histplot(ax=axes[0, 2], data=df, x='monthly_charges', hue='churned', kde=True, palette=['#4C72B0', '#C44E52'], multiple='stack')
    axes[0, 2].set_title("Monthly Charges Distribution by Churn", fontsize=15, weight='bold', pad=10)
    axes[0, 2].set_xlabel("Monthly Charges ($)", fontsize=12)
    axes[0, 2].set_ylabel("Count", fontsize=12)
    
    # 4. Contract type vs churn bar chart (Row 1, Col 0)
    contract_churn = df.groupby('contract_type')['churned'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    contract_churn.columns = ['contract_type', 'churn_rate_%']
    plt_sns.barplot(ax=axes[1, 0], data=contract_churn, x='contract_type', y='churn_rate_%', palette='viridis')
    axes[1, 0].set_title("Churn Rate by Contract Type", fontsize=15, weight='bold', pad=10)
    axes[1, 0].set_xlabel("Contract Type", fontsize=12)
    axes[1, 0].set_ylabel("Churn Rate (%)", fontsize=12)
    axes[1, 0].set_ylim(0, 100)
    for p in axes[1, 0].patches:
        axes[1, 0].annotate(f"{p.get_height():.1f}%", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 8), 
                            textcoords='offset points', weight='bold')
                            
    # 5. Feature importance chart (Row 1, Col 1)
    importances = results['feature_importances'].head(6) # top 6 features
    plt_sns.barplot(ax=axes[1, 1], x=importances.values, y=importances.index, palette='magma')
    axes[1, 1].set_title("Top Feature Importances (Decision Tree)", fontsize=15, weight='bold', pad=10)
    axes[1, 1].set_xlabel("Importance Score", fontsize=12)
    axes[1, 1].set_ylabel("Features", fontsize=12)
    
    # 6. Model accuracy comparison bar chart (Row 1, Col 2)
    models_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree'],
        'Accuracy (%)': [results['lr_accuracy'] * 100, results['dt_accuracy'] * 100]
    })
    plt_sns.barplot(ax=axes[1, 2], data=models_df, x='Model', y='Accuracy (%)', palette=['#3498DB', '#E67E22'])
    axes[1, 2].set_title("Model Accuracy Comparison", fontsize=15, weight='bold', pad=10)
    axes[1, 2].set_xlabel("Machine Learning Model", fontsize=12)
    axes[1, 2].set_ylabel("Test Accuracy (%)", fontsize=12)
    axes[1, 2].set_ylim(0, 100)
    for p in axes[1, 2].patches:
        axes[1, 2].annotate(f"{p.get_height():.2f}%", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 8), 
                            textcoords='offset points', weight='bold', fontsize=11)
                            
    plt.tight_layout()
    
    # Save dashboard precisely to outputs/dashboard.png
    out_dir = os.path.join(base_dir, "../outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated and saved Final Business Dashboard at: '{out_path}'")

if __name__ == "__main__":
    res = train_and_evaluate_models()
    if res:
        generate_final_dashboard(res)
