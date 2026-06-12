import os
import argparse
import numpy as np
import pandas as pd
from src.data import generate_synthetic_fraud_data, temporal_train_test_split
from src.cost_matrix import FinancialCostEvaluator
from src.models import (
    create_logistic_regression_pipeline, 
    create_smote_rf_pipeline, 
    create_cost_sensitive_rf_pipeline
)
from src.evaluate import evaluate_fraud_model
from src.visualize import plot_savings_threshold_curve, plot_model_savings_comparison

def run_project3(num_samples=10000, fraud_rate=0.005, seed=42):
    print("=" * 60)
    print("Starting Project 3: Cost-Sensitive Anomaly Detection")
    print("=" * 60)
    
    # 1. Generate fraud data
    print("Generating highly imbalanced synthetic fraud transaction dataset...")
    df = generate_synthetic_fraud_data(num_samples=num_samples, fraud_rate=fraud_rate, seed=seed)
    
    train_df, test_df = temporal_train_test_split(df, test_ratio=0.3)
    
    print(f"Total samples:       {len(df)}")
    print(f"Train set:          {len(train_df)} (Fraud: {train_df['is_fraud'].sum()} - {train_df['is_fraud'].mean():.2%})")
    print(f"Test set:           {len(test_df)} (Fraud: {test_df['is_fraud'].sum()} - {test_df['is_fraud'].mean():.2%})")
    
    # Columns definition
    high_card_cols = ["merchant_id", "device_id"]
    low_card_cols = ["card_type", "billing_country"]
    numerical_cols = ["amount"]
    
    # 2. Extract features and targets
    X_train = train_df.drop(columns=["is_fraud", "timestamp"])
    y_train = train_df["is_fraud"].values
    
    X_test = test_df.drop(columns=["is_fraud", "timestamp"])
    y_test = test_df["is_fraud"].values
    amounts_test = test_df["amount"].values
    
    # Initialize Business Evaluator
    evaluator = FinancialCostEvaluator(c_verify=10.0, c_irritate=25.0, c_fee=15.0)
    
    # 3. Train and evaluate models
    models_dict = {
        "Balanced Logistic Regression": create_logistic_regression_pipeline(
            high_card_cols, low_card_cols, numerical_cols, class_weight='balanced'
        ),
        "SMOTE + Random Forest": create_smote_rf_pipeline(
            high_card_cols, low_card_cols, numerical_cols
        ),
        "Cost-Sensitive Random Forest": create_cost_sensitive_rf_pipeline(
            high_card_cols, low_card_cols, numerical_cols
        )
    }
    
    results = {}
    optimal_savings_compare = {}
    
    for name, pipeline in models_dict.items():
        print(f"\nTraining model: {name}...")
        pipeline.fit(X_train, y_train)
        
        print(f"Evaluating {name}...")
        eval_metrics = evaluate_fraud_model(
            pipeline, X_test, y_test, amounts_test, evaluator=evaluator
        )
        results[name] = eval_metrics
        optimal_savings_compare[name] = eval_metrics["optimal_savings"]
        
    # 4. Display statistical vs financial metrics comparison
    print("\n" + "=" * 80)
    print("Model Evaluation Summary (Statistical vs. Financial Business KPIs):")
    print("=" * 80)
    print(f"{'Model Name':<30} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Opt Thresh':<10} | {'Opt Savings ($)':<15} | {'Opt Recall':<10}")
    print("-" * 88)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['roc_auc']:.4f}   | {metrics['pr_auc']:.4f}   | {metrics['optimal_threshold']:.4f}     | ${metrics['optimal_savings']:.2f}     | {metrics['optimal_recall']:.2%}")
    print("=" * 80)
    print(f"Baseline Fraud Loss (if all transactions approved): ${results['Cost-Sensitive Random Forest']['baseline_loss']:.2f}")
    
    # 5. Save visualizations
    print("\nGenerating evaluation plots...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot cost-curve for the best model (Cost-Sensitive Random Forest)
    best_model_name = "Cost-Sensitive Random Forest"
    best_metrics = results[best_model_name]
    plot_savings_threshold_curve(
        best_metrics["thresholds"],
        best_metrics["savings_curve"],
        best_metrics["optimal_threshold"],
        best_metrics["optimal_savings"],
        save_path=os.path.join(results_dir, "savings_threshold_curve.png")
    )
    print(f"Saved threshold savings curve to {results_dir}/savings_threshold_curve.png")
    
    # Plot model comparison chart
    plot_model_savings_comparison(
        optimal_savings_compare,
        save_path=os.path.join(results_dir, "model_savings_comparison.png")
    )
    print(f"Saved model savings comparison to {results_dir}/model_savings_comparison.png")
    
    print("\nProject 3 execution complete!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Number of transaction samples")
    parser.add_argument("--fraud_rate", type=float, default=0.005, help="Simulated fraud rate")
    args = parser.parse_args()
    
    run_project3(num_samples=args.samples, fraud_rate=args.fraud_rate)
