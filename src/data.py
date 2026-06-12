import numpy as np
import pandas as pd

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
