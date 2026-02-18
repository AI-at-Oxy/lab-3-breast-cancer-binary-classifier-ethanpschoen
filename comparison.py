"""
Model Comparison: Logistic Regression from Scratch vs. Random Forest

The Random Forest model was chosen due to its strong performance on binary classification tasks and its
ability to handle high-dimensional data with many features. As an ensemble method, it reduces overfitting
by averaging predictions across multiple decision trees, leading to better generalization on unseen data.
Random Forest also provides feature importance scores, which is valuable in a medical context where
understanding which tumor characteristics (e.g., radius, texture, concavity) are most predictive can offer
clinical insight. Additionally, it is robust to outliers and requires minimal preprocessing, making it
well-suited for this dataset without extensive feature scaling or transformation. Finally, it tends to
perform well out of the box, achieving high accuracy on the dataset even without significant hyperparameter tuning.
"""

from sklearn.ensemble import RandomForestClassifier
from binary_classification import train, predict, accuracy, load_data

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier on the training data.
    
    Args:
        X_train: (m, n) training feature matrix
        y_train: (m,) training labels
    
    Returns:
        trained Random Forest model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_random_forest(model, X_test):
    """
    Make predictions using the trained Random Forest model.
    
    Args:
        model: trained Random Forest model
        X_test: (m, n) test feature matrix
    
    Returns:
        y_pred: (m,) predicted labels
    """
    y_pred = model.predict(X_test)
    return y_pred

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Train
    w, b, losses = train(X_train, y_train, alpha=0.01, n_epochs=100)
    
    # Evaluate
    test_pred = predict(X_test, w, b)
    test_acc = accuracy(y_test, test_pred)
    
    print(f"From-scratch accuracy: {test_acc:.4f}")
    
    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    
    # Make predictions
    y_pred_rf = predict_random_forest(rf_model, X_test)
    
    # Calculate accuracy
    rf_accuracy = accuracy(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    # Compare results
    
    # The from-scratch gradient descent model slightly outperformed the Random Forest, achieving an accuracy of
    # 99.12% compared to 96.49%. This may be because logistic regression with gradient descent is well-suited
    # for linearly separable data, and the breast cancer dataset is known to be fairly linearly separable in its
    # feature space, allowing a linear decision boundary to classify cases with high precision. Random Forest,
    # while powerful, can sometimes underfit or introduce variance when ensemble parameters like the number of
    # trees or max depth aren't fully optimized, which may explain the small performance gap here.
