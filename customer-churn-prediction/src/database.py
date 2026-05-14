import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
try:
    from supabase import create_client, Client
except ImportError:
    Client = None

# Load environment variables from .env file
load_dotenv()

# =====================================================================
# 1. SUPABASE DATABASE SETUP SQL SCHEMA
# =====================================================================
# Run these SQL statements in your Supabase SQL Editor to create tables
# with proper primary keys, foreign keys, and check constraints.

SQL_SCHEMA_SETUP = """
-- Table 1: customers
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10) CHECK (gender IN ('Male', 'Female')),
    age INT CHECK (age >= 0),
    tenure INT CHECK (tenure >= 0),
    contract_type VARCHAR(50) CHECK (contract_type IN ('Month-to-month', 'One year', 'Two year')),
    monthly_charges NUMERIC(10, 2) CHECK (monthly_charges >= 0),
    total_charges NUMERIC(10, 2) CHECK (total_charges >= 0 OR total_charges IS NULL)
);

-- Table 2: services
CREATE TABLE services (
    customer_id VARCHAR(50) PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    phone_service VARCHAR(10) CHECK (phone_service IN ('Yes', 'No')),
    internet_service VARCHAR(50),
    streaming_tv VARCHAR(50),
    streaming_movies VARCHAR(50)
);

-- Table 3: churn_data
CREATE TABLE churn_data (
    customer_id VARCHAR(50) PRIMARY KEY REFERENCES customers(customer_id) ON DELETE CASCADE,
    churned VARCHAR(10) CHECK (churned IN ('Yes', 'No')),
    churn_reason TEXT,
    churn_date DATE
);
"""

# =====================================================================
# 2. SQL QUERIES (10 Queries requested)
# =====================================================================
SQL_QUERIES = {
    "1. Overall churn rate percentage": """
        SELECT ROUND(COUNT(*) FILTER (WHERE churned = 'Yes') * 100.0 / COUNT(*), 2) AS churn_rate_percentage
        FROM churn_data;
    """,
    "2. Average tenure of churned vs retained customers": """
        SELECT cd.churned, ROUND(AVG(c.tenure), 2) AS avg_tenure
        FROM customers c
        JOIN churn_data cd ON c.customer_id = cd.customer_id
        GROUP BY cd.churned;
    """,
    "3. Monthly revenue lost due to churn": """
        SELECT SUM(c.monthly_charges) AS monthly_revenue_lost
        FROM customers c
        JOIN churn_data cd ON c.customer_id = cd.customer_id
        WHERE cd.churned = 'Yes';
    """,
    "4. Churn rate by contract type": """
        SELECT c.contract_type,
               COUNT(*) AS total_customers,
               ROUND(COUNT(*) FILTER (WHERE cd.churned = 'Yes') * 100.0 / COUNT(*), 2) AS churn_rate_percentage
        FROM customers c
        JOIN churn_data cd ON c.customer_id = cd.customer_id
        GROUP BY c.contract_type;
    """,
    "5. Churn rate by internet service type": """
        SELECT s.internet_service,
               COUNT(*) AS total_customers,
               ROUND(COUNT(*) FILTER (WHERE cd.churned = 'Yes') * 100.0 / COUNT(*), 2) AS churn_rate_percentage
        FROM services s
        JOIN churn_data cd ON s.customer_id = cd.customer_id
        GROUP BY s.internet_service;
    """,
    "6. Top 5 reasons for churn": """
        SELECT churn_reason, COUNT(*) AS reason_count
        FROM churn_data
        WHERE churned = 'Yes' AND churn_reason IS NOT NULL
        GROUP BY churn_reason
        ORDER BY reason_count DESC
        LIMIT 5;
    """,
    "7. Customers with highest monthly charges who churned": """
        SELECT c.customer_id, c.monthly_charges, c.contract_type, cd.churn_reason
        FROM customers c
        JOIN churn_data cd ON c.customer_id = cd.customer_id
        WHERE cd.churned = 'Yes'
        ORDER BY c.monthly_charges DESC
        LIMIT 5;
    """,
    "8. Churn rate by gender": """
        SELECT c.gender,
               ROUND(COUNT(*) FILTER (WHERE cd.churned = 'Yes') * 100.0 / COUNT(*), 2) AS churn_rate_percentage
        FROM customers c
        JOIN churn_data cd ON c.customer_id = cd.customer_id
        GROUP BY c.gender;
    """,
    "9. Average monthly charges of churned vs retained": """
        SELECT cd.churned, ROUND(AVG(c.monthly_charges), 2) AS avg_monthly_charges
        FROM customers c
        JOIN churn_data cd ON c.customer_id = cd.customer_id
        GROUP BY cd.churned;
    """,
    "10. Total customers by contract type": """
        SELECT contract_type, COUNT(*) AS total_customers
        FROM customers
        GROUP BY contract_type;
    """
}

def get_supabase_client():
    """
    Initializes and returns the Supabase client using environment variables.
    Handles missing or placeholder configuration gracefully.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key or "your-project-id" in url:
        print("[Warning] Supabase URL or Key is not properly set in the .env file.")
        return None
        
    try:
        supabase: Client = create_client(url, key)
        return supabase
    except Exception as e:
        print(f"[Error] Failed to initialize Supabase client: {e}")
        return None

def insert_data_to_supabase(csv_path=None):
    """
    Reads the full CSV dataset, splits it into the 3 target tables,
    and inserts the records into Supabase using supabase-py.
    """
    if csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "../data/telco_churn.csv")
        
    supabase = get_supabase_client()
    if not supabase:
        print("[Info] Skipping direct Supabase insertion due to invalid client credentials.")
        return False
        
    try:
        print(f"Reading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Replace NaN with None for JSON/Supabase compatibility
        df = df.replace({np.nan: None})
        
        # Prepare customers data
        customers_df = df[['customer_id', 'gender', 'age', 'tenure', 'contract_type', 'monthly_charges', 'total_charges']]
        services_df = df[['customer_id', 'phone_service', 'internet_service', 'streaming_tv', 'streaming_movies']]
        churn_df = df[['customer_id', 'churned', 'churn_reason', 'churn_date']]
        
        print("Inserting records into 'customers' table...")
        # Batch insert using supabase-py
        customers_records = customers_df.to_dict(orient='records')
        supabase.table("customers").upsert(customers_records).execute()
        
        print("Inserting records into 'services' table...")
        services_records = services_df.to_dict(orient='records')
        supabase.table("services").upsert(services_records).execute()
        
        print("Inserting records into 'churn_data' table...")
        churn_records = churn_df.to_dict(orient='records')
        supabase.table("churn_data").upsert(churn_records).execute()
        
        print("Successfully inserted all dataset records into Supabase!")
        return True
    except Exception as e:
        print(f"[Error] An error occurred during database insertion: {e}")
        return False

def fetch_all_data(csv_fallback_path=None):
    """
    Fetches all data from the Supabase tables into pandas DataFrames,
    and merges them properly for analysis.
    If Supabase connection is not available, loads from the local CSV directly.
    """
    if csv_fallback_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_fallback_path = os.path.join(base_dir, "../data/telco_churn.csv")
        
    supabase = get_supabase_client()
    
    # Attempt to fetch from Supabase
    if supabase:
        try:
            print("Fetching data from Supabase tables...")
            cust_res = supabase.table("customers").select("*").execute()
            serv_res = supabase.table("services").select("*").execute()
            churn_res = supabase.table("churn_data").select("*").execute()
            
            if cust_res.data and len(cust_res.data) > 0:
                customers_df = pd.DataFrame(cust_res.data)
                services_df = pd.DataFrame(serv_res.data)
                churn_df = pd.DataFrame(churn_res.data)
                
                # Merge tables properly on customer_id
                merged_df = pd.merge(customers_df, services_df, on="customer_id", how="inner")
                merged_df = pd.merge(merged_df, churn_df, on="customer_id", how="inner")
                print("Successfully fetched and merged data from Supabase!")
                return merged_df
            else:
                print("[Info] Supabase tables are currently empty. Falling back to local data source.")
        except Exception as e:
            print(f"[Error] Failed to fetch data from Supabase: {e}. Falling back to local data source.")
            
    # Fallback to loading local CSV directly to keep workflow highly reliable
    try:
        print(f"Loading local merged dataset from '{csv_fallback_path}'...")
        merged_df = pd.read_csv(csv_fallback_path)
        return merged_df
    except Exception as e:
        print(f"[Critical Error] Could not load local dataset: {e}")
        return pd.DataFrame()

def run_10_queries_locally():
    """
    Executes the logic of the 10 requested SQL queries using pandas
    on the fetched/merged dataset to demonstrate outputs accurately.
    """
    df = fetch_all_data()
    if df.empty:
        print("Dataset is empty. Cannot run queries.")
        return
        
    print("\n" + "="*60)
    print("EXECUTION RESULTS OF THE 10 SQL QUERIES")
    print("="*60)
    
    # Query 1
    churn_rate = (df['churned'] == 'Yes').mean() * 100
    print(f"\n1. Overall churn rate percentage: {churn_rate:.2f}%")
    
    # Query 2
    avg_tenure = df.groupby('churned')['tenure'].mean()
    print(f"\n2. Average tenure of churned vs retained customers:\n{avg_tenure.to_string()}")
    
    # Query 3
    revenue_lost = df[df['churned'] == 'Yes']['monthly_charges'].sum()
    print(f"\n3. Monthly revenue lost due to churn: ${revenue_lost:,.2f}")
    
    # Query 4
    contract_churn = df.groupby('contract_type').apply(
        lambda x: pd.Series({
            'total_customers': len(x),
            'churn_rate_%': (x['churned'] == 'Yes').mean() * 100
        })
    )
    print(f"\n4. Churn rate by contract type:\n{contract_churn.to_string()}")
    
    # Query 5
    internet_churn = df.groupby('internet_service').apply(
        lambda x: pd.Series({
            'total_customers': len(x),
            'churn_rate_%': (x['churned'] == 'Yes').mean() * 100
        })
    )
    print(f"\n5. Churn rate by internet service type:\n{internet_churn.to_string()}")
    
    # Query 6
    reasons = df[df['churned'] == 'Yes']['churn_reason'].value_counts().head(5)
    print(f"\n6. Top 5 reasons for churn:\n{reasons.to_string()}")
    
    # Query 7
    top_churned_charges = df[df['churned'] == 'Yes'].sort_values(by='monthly_charges', ascending=False).head(5)
    print(f"\n7. Customers with highest monthly charges who churned:\n{top_churned_charges[['customer_id', 'monthly_charges', 'contract_type', 'churn_reason']].to_string(index=False)}")
    
    # Query 8
    gender_churn = df.groupby('gender').apply(
        lambda x: pd.Series({
            'churn_rate_%': (x['churned'] == 'Yes').mean() * 100
        })
    )
    print(f"\n8. Churn rate by gender:\n{gender_churn.to_string()}")
    
    # Query 9
    avg_charges = df.groupby('churned')['monthly_charges'].mean()
    print(f"\n9. Average monthly charges of churned vs retained:\n{avg_charges.to_string()}")
    
    # Query 10
    total_contract = df['contract_type'].value_counts()
    print(f"\n10. Total customers by contract type:\n{total_contract.to_string()}")
    print("\n" + "="*60)

if __name__ == "__main__":
    # Test file execution locally
    run_10_queries_locally()
