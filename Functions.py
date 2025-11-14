from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.svm import LinearSVC


def train_svm(x_train, y_train, x_val=None, y_val=None, x_test=None, y_test=None, C=1.0):
    """
    Train and evaluate an SVM classifier.

    Args:
        x_train (np.array): Training features
        y_train (np.array): Training labels
        x_val (np.array, optional): Validation features
        y_val (np.array, optional): Validation labels
        x_test (np.array, optional): Test features
        y_test (np.array, optional): Test labels
        kernel (str): SVM kernel ('linear', 'rbf', etc.)
        C (float): Regularization parameter
        gamma (str/float): Kernel coefficient
    Returns:
        clf: trained SVM classifier
        scaler: fitted scaler
    """
    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    
    if x_val is not None:
        x_val_scaled = scaler.transform(x_val)
    if x_test is not None:
        x_test_scaled = scaler.transform(x_test)
    
    # Create SVM classifier
    clf = LinearSVC(C=C, max_iter=10000, class_weight='balanced')
    clf.fit(x_train_scaled, y_train)
    
    # Evaluate on validation set
    if x_val is not None and y_val is not None:
        y_val_pred = clf.predict(x_val_scaled)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("Validation Classification Report:")
        print(classification_report(y_val, y_val_pred))
    
    # Evaluate on test set
    if x_test is not None and y_test is not None:
        y_test_pred = clf.predict(x_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_acc:.4f}")
        print("Test Classification Report:")
        print(classification_report(y_test, y_test_pred))
    
    return clf, scaler