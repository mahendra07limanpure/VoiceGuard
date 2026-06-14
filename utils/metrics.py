import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve

def calculate_metrics(y_true, y_prob, save_dir='assets', model_dir='model'):
    """
    Calculate and print all 5 required metrics for VoiceGuard.
    Saves confusion matrix plot and optimal threshold.
    """
    # Create output directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 4. EER and Optimal Threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    
    # Find the index where fpr and fnr are closest
    idx = np.argmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    optimal_threshold = thresholds[idx]
    
    # Save optimal_threshold to model/threshold.pkl
    threshold_path = os.path.join(model_dir, 'threshold.pkl')
    with open(threshold_path, 'wb') as f:
        pickle.dump(optimal_threshold, f)
    print(f"Saved optimal threshold {optimal_threshold:.4f} to {threshold_path}")
    
    # Apply optimal threshold to get binary predictions
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # 1. Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. F1 Score
    f1 = f1_score(y_true, y_pred)
    
    # 3. Per-Class Accuracy
    cm = confusion_matrix(y_true, y_pred)
    
    # Real is class 0, Fake is class 1
    # real_acc = TN / (TN + FP)
    # fake_acc = TP / (TP + FN)
    real_acc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    fake_acc = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    # Verify thresholds
    acc_pass = accuracy >= 0.80
    f1_pass = f1 >= 0.80
    eer_pass = eer <= 0.12
    real_pass = real_acc >= 0.75
    fake_pass = fake_acc >= 0.75
    
    overall_pass = acc_pass and f1_pass and eer_pass and real_pass and fake_pass
    
    # Checkmarks
    c_acc = "✓" if acc_pass else "✗"
    c_f1 = "✓" if f1_pass else "✗"
    c_eer = "✓" if eer_pass else "✗"
    c_real = "✓" if real_pass else "✗"
    c_fake = "✓" if fake_pass else "✗"
    
    status_str = "PASS" if overall_pass else "FAIL"
    
    # Print formatted report
    print("╔══════════════════════════════════════╗")
    print("║       VOICEGUARD METRICS REPORT      ║")
    print("╠══════════════════════════════════════╣")
    print(f"║ Overall Accuracy    :   {accuracy*100:5.2f}%  {c_acc}  ║")
    print(f"║ F1 Score            :   {f1*100:5.2f}%  {c_f1}  ║")
    print(f"║ EER                 :   {eer*100:5.2f}%  {c_eer}  ║")
    print(f"║ Genuine Accuracy    :   {real_acc*100:5.2f}%  {c_real}  ║")
    print(f"║ Deepfake Accuracy   :   {fake_acc*100:5.2f}%  {c_fake}  ║")
    print("╠══════════════════════════════════════╣")
    print(f"║ VERIFICATION STATUS :   {status_str:4}         ║")
    print("╚══════════════════════════════════════╝")
    
    # 5. Confusion Matrix Heatmap (dark theme)
    plt.style.use('dark_background')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Deepfake'],
                yticklabels=['Genuine', 'Deepfake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('VoiceGuard Confusion Matrix')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'eer': eer,
        'genuine_accuracy': real_acc,
        'deepfake_accuracy': fake_acc,
        'verification_status': status_str,
        'optimal_threshold': optimal_threshold
    }

def save_norm_params(X_train, model_dir='model'):
    """
    Compute global mean and std from training features and save to norm_params.pkl.
    X_train shape: (N, 128, 128, 2)
    """
    os.makedirs(model_dir, exist_ok=True)
    # Compute mean and std for each channel independently across all samples
    mean0 = float(np.mean(X_train[:, :, :, 0]))
    std0 = float(np.std(X_train[:, :, :, 0]))
    mean1 = float(np.mean(X_train[:, :, :, 1]))
    std1 = float(np.std(X_train[:, :, :, 1]))
    
    norm_params = {
        'mean': [mean0, mean1],
        'std': [std0, std1]
    }
    
    norm_path = os.path.join(model_dir, 'norm_params.pkl')
    with open(norm_path, 'wb') as f:
        pickle.dump(norm_params, f)
    print(f"Saved norm params to {norm_path}: {norm_params}")
    return norm_params
