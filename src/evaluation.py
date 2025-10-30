"""
Evaluation utilities for anomaly detection performance assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Tuple, Dict, Any, Optional, List

def calculate_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for anomaly detection.
    
    Args:
        y_true: Ground truth binary labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores (higher = more anomalous)
        threshold: Optional threshold for binary classification
        
    Returns:
        metrics: Dictionary containing various performance metrics
    """
    # ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve and AUC
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'roc_thresholds': roc_thresholds,
        'pr_thresholds': pr_thresholds
    }
    
    # If threshold provided, calculate binary classification metrics
    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Confusion matrix components
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        metrics.update({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
    
    return metrics

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path: str = None):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_synthetic_ground_truth(cube_shape: Tuple[int, int, int], anomaly_percentage: float = 0.05) -> np.ndarray:
    """
    Create synthetic ground truth for demonstration purposes.
    
    Args:
        cube_shape: Shape of the hyperspectral cube (height, width, bands)
        anomaly_percentage: Percentage of pixels to mark as anomalous
        
    Returns:
        ground_truth: Binary ground truth array (height, width)
    """
    height, width, _ = cube_shape
    
    # Create random anomalies
    total_pixels = height * width
    n_anomalies = int(total_pixels * anomaly_percentage)
    
    # Initialize ground truth
    ground_truth = np.zeros((height, width), dtype=np.uint8)
    
    # Add random anomalies
    anomaly_indices = np.random.choice(total_pixels, size=n_anomalies, replace=False)
    anomaly_coords = np.unravel_index(anomaly_indices, (height, width))
    ground_truth[anomaly_coords] = 1
    
    print(f"Created synthetic ground truth: {n_anomalies} anomalies ({anomaly_percentage*100:.1f}%)")
    
    return ground_truth

def evaluate_detection(
    distances: np.ndarray,
    ground_truth: np.ndarray,
    k_values: List[float],
    median_dist: float,
    mad_dist: float,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of anomaly detection performance.
    
    Args:
        distances: Mahalanobis distances array
        ground_truth: Binary ground truth array  
        k_values: List of k values for threshold evaluation
        median_dist: Median distance for thresholding
        mad_dist: MAD of distances for thresholding
        save_path: Optional path to save evaluation plots
        
    Returns:
        results: Dictionary containing evaluation results
    """
    # Flatten arrays and remove invalid values
    distances_flat = distances.flatten()
    gt_flat = ground_truth.flatten()
    
    # Remove NaN and infinite values
    valid_mask = np.isfinite(distances_flat)
    distances_valid = distances_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]
    
    if len(distances_valid) == 0:
        return {'error': 'No valid distance values found'}
    
    # Overall performance (using all valid distances as scores)
    overall_metrics = calculate_metrics(gt_valid, distances_valid)
    
    # Per-k evaluation
    k_results = {}
    
    for k in k_values:
        threshold = median_dist + k * mad_dist
        k_metrics = calculate_metrics(gt_valid, distances_valid, threshold)
        k_results[f'k_{k}'] = k_metrics
    
    # Find best performance
    best_f1 = 0
    best_k = k_values[0]
    
    for k in k_values:
        f1 = k_results[f'k_{k}'].get('f1_score', 0)
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
    
    # Compile results
    results = {
        'overall': overall_metrics,
        'k_results': k_results,
        'best_k': best_k,
        'best_f1': best_f1,
        'median_dist': median_dist,
        'mad_dist': mad_dist,
        'n_valid_pixels': len(distances_valid),
        'n_anomalies_gt': np.sum(gt_valid)
    }
    
    # Create evaluation plots
    if save_path:
        create_evaluation_plots(results, save_path)
    
    return results

def create_evaluation_plots(results: Dict[str, Any], save_path: str):
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    fpr = results['overall']['fpr']
    tpr = results['overall']['tpr']
    roc_auc = results['overall']['roc_auc']
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    
    # Precision-Recall Curve
    precision = results['overall']['precision']
    recall = results['overall']['recall']
    pr_auc = results['overall']['pr_auc']
    
    axes[0, 1].plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    
    # F1-Score vs k values
    k_values = [float(k.split('_')[1]) for k in results['k_results'].keys()]
    f1_scores = [results['k_results'][k]['f1_score'] for k in results['k_results'].keys()]
    
    axes[1, 0].plot(k_values, f1_scores, 'bo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('k value')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score vs k values')
    axes[1, 0].grid(True)
    
    # Accuracy vs k values
    accuracies = [results['k_results'][k]['accuracy'] for k in results['k_results'].keys()]
    
    axes[1, 1].plot(k_values, accuracies, 'ro-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('k value')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs k values')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to: {save_path}")