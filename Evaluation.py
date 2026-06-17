import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curves(
    model,
    X_test,
    y_test,
    model_name="XGBoost",
    n_bins=20,
    class_labels=None,
    figsize=(12, 5)
):
    """
    Plot calibration curves (reliability diagrams) for multiclass model.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels (should be 0, 1, 2 for multiclass)
        model_name: Name for display
        n_bins: Number of bins for calibration curve
        class_labels: List of class names, e.g., ['true_match', 'wrong_match', 'fake']
        figsize: Figure size
    
    Returns:
        dict with calibration metrics and figure
    """
    
    # Get predictions
    X_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    proba = model.predict_proba(X_array)  # Shape: (n_samples, n_classes)
    
    n_classes = proba.shape[1]
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    
    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    if n_classes == 1:
        axes = [axes]
    
    results = {}
    
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        
        # One-vs-rest labels for this class
        y_binary = (y_test == class_idx).astype(int)
        proba_class = proba[:, class_idx]
        
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            y_binary, proba_class, n_bins=n_bins, strategy='uniform'
        )
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
        ax.plot(prob_pred, prob_true, 'o-', label=model_name, linewidth=2, markersize=8, color='steelblue')
        
        ax.fill_between([0, 1], [0, 1], [0.5, 0.5], alpha=0.1, color='green', label='Well Calibrated Zone')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax.set_ylabel('Fraction of Positives (True Frequency)', fontsize=11)
        ax.set_title(f'{class_labels[class_idx]} (1-vs-Rest)', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        # Compute Expected Calibration Error (ECE)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        results[class_labels[class_idx]] = {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ECE': ece
        }
        
        print(f"{class_labels[class_idx]:20s} | ECE: {ece:.4f}")
    
    fig.suptitle(f'Calibration Curves: {model_name}', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('calibration_curves.png', dpi=100, bbox_inches='tight')
    print("\n✓ Saved: calibration_curves.png")
    
    return results, fig


def compute_ece(y_true, y_proba, n_bins=10):
    """
    Expected Calibration Error: mean absolute deviation between predicted 
    and true frequencies across probability bins.
    
    Lower is better (0 = perfect calibration).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    return np.mean(np.abs(prob_true - prob_pred))