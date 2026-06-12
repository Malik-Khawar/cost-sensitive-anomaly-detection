import matplotlib.pyplot as plt
import numpy as np
import os

def setup_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'grid.color': '#2A2A2A',
        'grid.linestyle': '--',
        'axes.facecolor': '#1E1E1E',
        'figure.facecolor': '#121212',
        'text.color': '#E0E0E0',
        'axes.labelcolor': '#CCCCCC',
        'xtick.color': '#888888',
        'ytick.color': '#888888',
        'font.family': 'sans-serif',
        'font.size': 10,
        'figure.autolayout': True
    })

def plot_savings_threshold_curve(thresholds, savings_curve, optimal_threshold, optimal_savings, save_path=None):
    """
    Plots Net Financial Savings vs. Decision Threshold.
    """
    setup_dark_theme()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot curve
    ax.plot(thresholds, savings_curve, color="#54A0FF", linewidth=2.5, label="Net Financial Savings")
    
    # Highlight optimal threshold
    ax.scatter(optimal_threshold, optimal_savings, color="#10AC84", s=100, zorder=5)
    ax.axvline(optimal_threshold, color="#10AC84", linestyle="--", alpha=0.7, 
               label=f"Optimal Threshold ({optimal_threshold:.3f})")
    
    # Baseline line (savings = 0)
    ax.axhline(0.0, color="#FF9F43", linestyle=":", label="Baseline (Approved All)")
    
    # Styling
    ax.set_title("Expected Monetary Value (EMV) Savings vs. Threshold", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Fraud Probability Threshold", fontsize=10)
    ax.set_ylabel("Net Savings ($)", fontsize=10)
    ax.grid(True)
    ax.legend(loc="lower center", framealpha=0.8, facecolor="#1A1A1A", edgecolor="#333333")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_model_savings_comparison(model_savings_dict, save_path=None):
    """
    Plots a bar chart comparing the total dollar savings of multiple models on test set.
    """
    setup_dark_theme()
    
    models = list(model_savings_dict.keys())
    savings = list(model_savings_dict.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Horizontal bar chart
    bars = ax.barh(models, savings, color=["#FF9F43", "#54A0FF", "#10AC84"], edgecolor="#333333")
    
    # Add values on top of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(savings)*0.01, bar.get_y() + bar.get_height()/2, 
                f"${width:,.2f}", 
                va='center', ha='left', color='#FFFFFF', fontweight='bold')
                
    ax.set_title("Model Net Financial Savings Comparison", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Net Savings ($) vs. Approval Baseline")
    ax.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
