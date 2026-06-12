import numpy as np

class FinancialCostEvaluator:
    """
    Computes business-oriented financial savings metrics using custom utility matrices.
    """
    def __init__(self, c_verify=10.0, c_irritate=25.0, c_fee=15.0):
        self.c_verify = c_verify      # Cost to call/verify a suspected fraud (TP)
        self.c_irritate = c_irritate  # Cost of blocking legitimate user (FP)
        self.c_fee = c_fee            # Chargeback/processing fee on missed fraud (FN)

    def compute_utilities(self, y_true, y_pred_binary, amounts):
        """
        Computes the net financial utility for each transaction.
        Args:
            y_true: Ground truth binary array (1=fraud, 0=legitimate)
            y_pred_binary: Predicted binary array (1=blocked/suspected, 0=approved)
            amounts: Series or array of transaction amounts
        """
        amounts = np.array(amounts)
        y_true = np.array(y_true)
        y_pred_binary = np.array(y_pred_binary)
        
        # Net utility arrays
        utilities = np.zeros_like(amounts)
        
        # TP: Correctly blocked fraud
        # Utility = -verification cost (we avoid losing the fraud amount)
        tp_mask = (y_true == 1) & (y_pred_binary == 1)
        utilities[tp_mask] = -self.c_verify
        
        # FP: Legitimate transaction blocked
        # Utility = -irritation cost (verification cost + admin cost)
        fp_mask = (y_true == 0) & (y_pred_binary == 1)
        utilities[fp_mask] = -self.c_irritate
        
        # FN: Missed fraud (fraud approved)
        # Utility = - (transaction amount + chargeback fee)
        fn_mask = (y_true == 1) & (y_pred_binary == 0)
        utilities[fn_mask] = -(amounts[fn_mask] + self.c_fee)
        
        # TN: Legitimate transaction approved
        # Utility = 0
        tn_mask = (y_true == 0) & (y_pred_binary == 0)
        utilities[tn_mask] = 0.0
        
        return utilities

    def compute_net_savings(self, y_true, y_pred_binary, amounts):
        """
        Net Savings = (Model Utility) - (Baseline Utility: 'Do Nothing' where all txns approved)
        """
        # Model utility
        model_utility = np.sum(self.compute_utilities(y_true, y_pred_binary, amounts))
        
        # Baseline utility (all y_pred_binary = 0)
        baseline_pred = np.zeros_like(y_true)
        baseline_utility = np.sum(self.compute_utilities(y_true, baseline_pred, amounts))
        
        # Net savings
        net_savings = model_utility - baseline_utility
        return net_savings, model_utility, baseline_utility

    def find_optimal_threshold(self, y_true, y_probs, amounts, steps=100):
        """
        Grid search to find the probability threshold that maximizes net savings.
        """
        thresholds = np.linspace(0.0001, 0.9999, steps)
        best_savings = -np.inf
        best_threshold = 0.5
        
        savings_curve = []
        
        for thresh in thresholds:
            y_pred_binary = (y_probs >= thresh).astype(int)
            savings, _, _ = self.compute_net_savings(y_true, y_pred_binary, amounts)
            savings_curve.append(savings)
            
            if savings > best_savings:
                best_savings = savings
                best_threshold = thresh
                
        return best_threshold, best_savings, thresholds, np.array(savings_curve)
