import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_telco_data(num_rows=1000, output_path="../data/telco_churn.csv"):
    """
    Generates a realistic Telco Customer Churn dataset matching the required schema:
    customers: customer_id, gender, age, tenure, contract_type, monthly_charges, total_charges
    services: customer_id, phone_service, internet_service, streaming_tv, streaming_movies
    churn_data: customer_id, churned, churn_reason, churn_date
    """
    np.random.seed(42)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, num_rows + 1)]
    genders = np.random.choice(['Male', 'Female'], size=num_rows)
    ages = np.random.randint(18, 80, size=num_rows)
    
    # Contract types and baseline churn probabilities
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_rows, p=[0.55, 0.25, 0.20])
    
    tenure = []
    monthly_charges = []
    churned = []
    churn_reasons = []
    churn_dates = []
    
    phone_services = np.random.choice(['Yes', 'No'], size=num_rows, p=[0.9, 0.1])
    internet_services = []
    streaming_tvs = []
    streaming_movies = []
    
    reasons_list = [
        "Competitor offered better devices",
        "Competitor made better offer",
        "Attitude of support person",
        "Don't know",
        "Price too high",
        "Network reliability",
        "Poor customer service"
    ]
    
    base_date = datetime(2025, 1, 1)
    
    for i in range(num_rows):
        c_type = contract_types[i]
        
        # Tenure depends on contract type
        if c_type == 'Month-to-month':
            t = np.random.randint(1, 24)
            churn_prob = 0.45
        elif c_type == 'One year':
            t = np.random.randint(12, 60)
            churn_prob = 0.15
        else:
            t = np.random.randint(24, 73)
            churn_prob = 0.05
            
        # Services
        i_service = np.random.choice(['DSL', 'Fiber optic', 'No'], p=[0.35, 0.45, 0.20])
        internet_services.append(i_service)
        
        if i_service == 'No':
            s_tv = 'No internet service'
            s_mov = 'No internet service'
            m_charge = np.random.uniform(18.0, 25.0)
        else:
            s_tv = np.random.choice(['Yes', 'No'])
            s_mov = np.random.choice(['Yes', 'No'])
            if i_service == 'Fiber optic':
                m_charge = np.random.uniform(70.0, 115.0)
                churn_prob += 0.1  # Higher cost fiber optic has slightly higher churn
            else:
                m_charge = np.random.uniform(45.0, 70.0)
                
        streaming_tvs.append(s_tv)
        streaming_movies.append(s_mov)
                
        if phone_services[i] == 'Yes':
            m_charge += 10.0
            
        tenure.append(t)
        monthly_charges.append(round(m_charge, 2))
        
        # Higher monthly charges and lower tenure increase churn risk
        if t < 6:
            churn_prob += 0.15
            
        is_churned = 'Yes' if np.random.rand() < churn_prob else 'No'
        churned.append(is_churned)
        
        if is_churned == 'Yes':
            churn_reasons.append(np.random.choice(reasons_list))
            # Random churn date within the last few months
            c_date = base_date - timedelta(days=np.random.randint(1, 180))
            churn_dates.append(c_date.strftime('%Y-%m-%d'))
        else:
            churn_reasons.append(np.nan)
            churn_dates.append(np.nan)
            
    # Calculate total charges with some intentional missing values for tenure = 0 or brand new
    total_charges = []
    for t, m in zip(tenure, monthly_charges):
        if t == 0:
            total_charges.append(np.nan)
        else:
            # Add a small random variation
            tc = t * m * np.random.uniform(0.95, 1.05)
            total_charges.append(round(tc, 2))
            
    # Introduce a couple of random missing values to test preprocessing robustness
    for idx in np.random.choice(range(num_rows), size=5, replace=False):
        if tenure[idx] != 0:
            total_charges[idx] = np.nan
            
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': genders,
        'age': ages,
        'tenure': tenure,
        'contract_type': contract_types,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'phone_service': phone_services,
        'internet_service': internet_services,
        'streaming_tv': streaming_tvs,
        'streaming_movies': streaming_movies,
        'churned': churned,
        'churn_reason': churn_reasons,
        'churn_date': churn_dates
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_rows} rows of Telco data at '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_telco_data(1000, "c:/Users/shrid/Downloads/people chrun/customer-churn-prediction/data/telco_churn.csv")
