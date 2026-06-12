import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, classification_report
from src.cost_matrix import FinancialCostEvaluator

def evaluate_fraud_model(model, X_test, y_test, amounts, evaluator=None, threshold=0.5):
    """
    Evaluates a trained model pipeline on standard statistical metrics and custom
    financial utility metrics.
    """
    if evaluator is None:
        evaluator = FinancialCostEvaluator()
        
    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Predict default binary labels
    y_pred_binary = (y_probs >= threshold).astype(int)
    
    # Standard metrics
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # Find optimal threshold to maximize savings
    best_thresh, best_savings, thresholds, savings_curve = evaluator.find_optimal_threshold(
        y_test, y_probs, amounts
    )
    
    # Evaluate model at its own default threshold
    savings, model_utility, baseline_utility = evaluator.compute_net_savings(
        y_test, y_pred_binary, amounts
    )
    
    # Evaluate model at the optimal threshold
    y_pred_opt = (y_probs >= best_thresh).astype(int)
    opt_savings, opt_model_utility, _ = evaluator.compute_net_savings(
        y_test, y_pred_opt, amounts
    )
    
    # Precision and recall at default and optimal thresholds
    precision_default = np.mean(y_test[y_pred_binary == 1] == 1) if np.sum(y_pred_binary) > 0 else 0.0
    recall_default = np.mean(y_pred_binary[y_test == 1] == 1)
    
    precision_opt = np.mean(y_test[y_pred_opt == 1] == 1) if np.sum(y_pred_opt) > 0 else 0.0
    recall_opt = np.mean(y_pred_opt[y_test == 1] == 1)
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "default_threshold": float(threshold),
        "default_savings": float(savings),
        "default_precision": float(precision_default),
        "default_recall": float(recall_default),
        "optimal_threshold": float(best_thresh),
        "optimal_savings": float(opt_savings),
        "optimal_precision": float(precision_opt),
        "optimal_recall": float(recall_opt),
        "baseline_loss": float(-baseline_utility),
        "y_probs": y_probs,
        "thresholds": thresholds,
        "savings_curve": savings_curve
    }
