import numpy as np
import pandas as pd


def load_real_fraud_data():
    """
    Loads the real Credit Card Fraud Detection dataset from OpenML.
    Dataset: 284,807 European credit card transactions from Sept 2013.
    Features: V1-V28 (PCA-transformed), Time, Amount.
    Target: Class (0=Legitimate, 1=Fraud). Fraud rate: 0.17%.
    """
    from sklearn.datasets import fetch_openml

    print("Downloading Credit Card Fraud dataset from OpenML (first run may take a minute)...")
    data = fetch_openml(name='creditcard', version=1, as_frame=True, parser='auto')
    df = data.frame

    print(f"  Raw dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Build a case-insensitive column mapping for robustness
    col_map = {c.lower(): c for c in df.columns}

    # Rename target column → is_fraud
    target_col = col_map.get('class')
    if target_col and target_col != 'is_fraud':
        df = df.rename(columns={target_col: 'is_fraud'})
    df['is_fraud'] = df['is_fraud'].astype(int)

    # Rename Amount → amount
    amount_col = col_map.get('amount')
    if amount_col and amount_col != 'amount':
        df = df.rename(columns={amount_col: 'amount'})

    # Rename Time → timestamp
    time_col = col_map.get('time')
    if time_col:
        df = df.rename(columns={time_col: 'timestamp'})
    else:
        # If no Time column exists, create a sequential timestamp
        print("  Warning: No 'Time' column found; creating sequential timestamps.")
        df['timestamp'] = np.arange(len(df), dtype=float)

    return df


def generate_synthetic_fraud_data(num_samples=10000, fraud_rate=0.005, seed=42):
    """
    Generates a highly imbalanced credit card fraud transaction dataset
    with high-cardinality categorical variables.
    """
    np.random.seed(seed)
    
    # Generate timestamp order
    time_deltas = np.random.exponential(scale=60, size=num_samples) # average 1 min between txns
    timestamps = np.cumsum(time_deltas)
    
    # Transaction amount: log-normal distribution (often heavy-tailed)
    amounts = np.random.lognormal(mean=3.5, sigma=1.0, size=num_samples)
    
    # Categorical variables
    card_types = ["visa", "mastercard", "amex", "discover"]
    card_probs = [0.5, 0.35, 0.1, 0.05]
    cards = np.random.choice(card_types, size=num_samples, p=card_probs)
    
    countries = ["US", "CA", "GB", "DE", "FR", "IN", "CN", "BR"]
    country_probs = [0.7, 0.08, 0.05, 0.05, 0.03, 0.04, 0.03, 0.02]
    billing_countries = np.random.choice(countries, size=num_samples, p=country_probs)
    
    # High-cardinality features
    num_merchants = 400
    merchants = [f"M_{i:03d}" for i in range(num_merchants)]
    # Power-law distribution for merchants (some merchants are very common, others rare)
    merchant_weights = 1.0 / (np.arange(1, num_merchants + 1) ** 0.8)
    merchant_weights /= merchant_weights.sum()
    merchant_ids = np.random.choice(merchants, size=num_samples, p=merchant_weights)
    
    num_devices = 300
    devices = [f"D_{i:03d}" for i in range(num_devices)]
    device_weights = 1.0 / (np.arange(1, num_devices + 1) ** 0.9)
    device_weights /= device_weights.sum()
    device_ids = np.random.choice(devices, size=num_samples, p=device_weights)
    
    # Determine fraud labels (imbalanced)
    num_fraud = int(num_samples * fraud_rate)
    is_fraud = np.zeros(num_samples, dtype=int)
    
    # Fraud rule logic:
    # 1. Extremely large transaction amounts are more likely to be fraud.
    # 2. Specific merchants and device IDs are compromised (higher fraud probability).
    # 3. Amex card type + foreign countries has slightly higher fraud.
    
    # Calculate raw logits for fraud
    # Normalize amounts for logit computation
    amt_norm = (amounts - np.mean(amounts)) / np.std(amounts)
    
    # Compromised merchants and devices
    compromised_merchants = set(np.random.choice(merchants, size=15, replace=False))
    compromised_devices = set(np.random.choice(devices, size=10, replace=False))
    
    logits = -6.0 + 1.2 * amt_norm
    for i in range(num_samples):
        if merchant_ids[i] in compromised_merchants:
            logits[i] += 3.5
        if device_ids[i] in compromised_devices:
            logits[i] += 4.0
        if cards[i] == "amex" and billing_countries[i] not in ["US", "CA"]:
            logits[i] += 1.5
            
    # Sigmoid to get probability
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    # Select exactly the top target fraud cases to match the fraud rate
    fraud_indices = np.argsort(probs)[-num_fraud:]
    is_fraud[fraud_indices] = 1
    
    # Let's adjust transaction amounts for fraudulent transactions to make it highly heavy-tailed
    # Fraud transactions usually have much higher amounts than normal ones on average
    amounts[is_fraud == 1] *= np.random.uniform(1.5, 4.0, size=num_fraud)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "amount": amounts,
        "card_type": cards,
        "billing_country": billing_countries,
        "merchant_id": merchant_ids,
        "device_id": device_ids,
        "is_fraud": is_fraud
    })
    
    return df

def temporal_train_test_split(df, test_ratio=0.25):
    """
    Splits the fraud dataset based on time (timestamp) to represent real production setups.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    
    train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df


def load_real_fraud_data():
    """
    Loads the real Credit Card Fraud Detection dataset from OpenML.
    Dataset: 284,807 European credit card transactions from Sept 2013.
    Features: V1-V28 (PCA-transformed), Amount.
    Target: Class (0=Legitimate, 1=Fraud). Fraud rate: 0.17%.
    """
    from sklearn.datasets import fetch_openml
    import numpy as np
    
    print("Downloading Credit Card Fraud dataset from OpenML (first run may take a minute)...")
    data = fetch_openml(name='creditcard', version=1, as_frame=True, parser='auto')
    df = data.frame
    
    # Rename columns to match our convention
    df = df.rename(columns={'Class': 'is_fraud'})
    df['is_fraud'] = df['is_fraud'].astype(int)
    
    # Rename 'Amount' to 'amount' for consistency
    df = df.rename(columns={'Amount': 'amount'})
    
    # The OpenML version 1 drops the 'Time' column, so we add a synthetic timestamp
    # to maintain compatibility with our temporal_train_test_split.
    df['timestamp'] = np.arange(len(df))
    
    return df


