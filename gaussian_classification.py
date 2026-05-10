"""
Gaussian Classification Model Project
Train and Improve a Gaussian Classification Model

This project uses Gaussian Naive Bayes on the Iris dataset.
It trains a baseline model, improves preprocessing using StandardScaler,
evaluates accuracy, confusion matrix, and classification report,
and saves the trained model.
"""

import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("Dataset: Iris Flower Dataset")
    print("Features:", iris.feature_names)
    print("Classes:", iris.target_names)
    print("Total samples:", len(X))

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Baseline Gaussian Naive Bayes model
    baseline_model = GaussianNB()
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    print("\nBaseline GaussianNB Accuracy:", round(baseline_accuracy, 4))

    # 4. Improved model using preprocessing pipeline
    improved_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ])

    improved_model.fit(X_train, y_train)
    improved_preds = improved_model.predict(X_test)
    improved_accuracy = accuracy_score(y_test, improved_preds)

    print("Improved GaussianNB Accuracy:", round(improved_accuracy, 4))

    # 5. Cross-validation for stronger evaluation
    cv_scores = cross_val_score(improved_model, X, y, cv=5)
    print("\n5-Fold Cross Validation Scores:", cv_scores)
    print("Mean CV Accuracy:", round(cv_scores.mean(), 4))

    # 6. Classification report
    report = classification_report(y_test, improved_preds, target_names=iris.target_names)
    print("\nClassification Report:")
    print(report)

    # Save report to text file
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write("Gaussian Classification Model Report\n")
        f.write("====================================\n\n")
        f.write(f"Baseline Accuracy: {baseline_accuracy:.4f}\n")
        f.write(f"Improved Accuracy: {improved_accuracy:.4f}\n")
        f.write(f"Mean Cross Validation Accuracy: {cv_scores.mean():.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # 7. Confusion matrix visualization
    cm = confusion_matrix(y_test, improved_preds)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    display.plot()
    plt.title("Gaussian Classification Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 8. Save trained model
    joblib.dump(improved_model, output_dir / "gaussian_classifier_model.pkl")

    print("\nFiles saved in outputs folder:")
    print("- classification_report.txt")
    print("- confusion_matrix.png")
    print("- gaussian_classifier_model.pkl")


if __name__ == "__main__":
    main()