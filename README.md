# Gaussian Classification Model Project

## Project Title
Train and Improve Your Model for Gaussian Classification Model

## Objective
The goal of this project is to train a Gaussian Classification model and improve its performance using proper preprocessing and evaluation methods.

## Model Used
This project uses **Gaussian Naive Bayes**, a classification algorithm based on probability and the normal/Gaussian distribution.

## Dataset
The project uses the Iris dataset from scikit-learn. The dataset contains flower measurements and class labels.

## Steps Completed
1. Loaded the Iris dataset
2. Split the dataset into training and testing data
3. Trained a baseline Gaussian Naive Bayes classifier
4. Improved the model using StandardScaler preprocessing
5. Evaluated the model using:
   - Accuracy
   - Cross-validation
   - Confusion matrix
   - Classification report
6. Saved the trained model as a `.pkl` file

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python gaussian_classification.py
```

## Output Files
After running the code, the `outputs` folder will contain:

- `classification_report.txt`
- `confusion_matrix.png`
- `gaussian_classifier_model.pkl`

## Conclusion
The Gaussian Classification model performed well on the Iris dataset. Using preprocessing and cross-validation helped improve the reliability of the model evaluation. The final model can classify flower types based on input measurements.