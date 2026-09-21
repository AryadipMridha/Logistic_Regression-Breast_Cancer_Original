# Predicting Breast Cancer with Logistic Regression

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)](https://scikit-learn.org/stable/)

This project classifies breast tumours as benign or malignant from cell measurements, using logistic regression on the Breast Cancer Wisconsin (Original) dataset from the UCI Machine Learning Repository.

| Metric | Result |
|---|---|
| 10-fold cross-validation accuracy (training split) | **96.60 %** (standard deviation 2.58 %) |
| Held-out test accuracy | **97.14 %** (136 of 140 correct) |

## The "what" and "why" of logistic regression

Imagine you want to answer a yes-or-no question. Is this email spam? Will this customer churn? Is this tumour malignant?

Logistic regression is a natural fit for this kind of question. It takes a set of inputs (clump thickness, cell size and so on) and estimates the *probability* of a "yes".

- A linear combination of the features, $z = w^\top x + b$, is passed through the **sigmoid (logistic) function**. The sigmoid squashes any real number into a smooth S-shaped curve between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- A threshold then turns the probability into a label. The default is 0.5: above it the model predicts malignant, below it benign.

The algorithm is fast and interpretable, and it makes an excellent baseline for any classification task.

---

## Project pipeline

1. **Data acquisition.** The dataset is fetched programmatically from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast-cancer-wisconsin-original) with the `ucimlrepo` library. It has 9 features, and the target uses Class 2 for benign and Class 4 for malignant.

2. **Preprocessing.** The `Bare_nuclei` column has 16 missing values (`NaN`). They are filled with the column mean using scikit-learn's `SimpleImputer`.

3. **Train/test split.** 20% of the data (140 samples) is held out as a test set with `random_state=0`. The model never sees these samples during training.

4. **Model training.** A `LogisticRegression` classifier (`random_state=0`, default settings otherwise) is fitted on the 80% training split. It learns one weight per feature to separate benign from malignant tumours.

5. **Evaluation.** Two complementary checks are used:
    * **A confusion matrix on the held-out test set** shows exactly which predictions were right and wrong (see below).
    * **10-fold cross-validation on the training split** (`cv=10`) repeats training and validation on 10 different folds. The mean and standard deviation show how stable the model's performance is.

---

## Feature scaling: does it help?

The usual advice is to standardise features (for example with `StandardScaler`) before fitting a linear model, so that features with large ranges don't dominate. This was tested with 10-fold cross-validation on the training split:

| Setup | CV accuracy | Standard deviation |
|---|---:|---:|
| Without feature scaling | **96.60 %** | 2.58 % |
| With `StandardScaler` | 96.43 % | 2.52 % |

**Conclusion.** The 0.17-point gap is far smaller than the fold-to-fold standard deviation of about 2.5 points, so scaling makes no meaningful difference here. That is expected: all nine features are integer scores on the same 1–10 scale, so no feature dominates the others by magnitude. The final model is kept unscaled because it is simpler. The general lesson still holds: test assumptions instead of applying best practices blindly.

---

## Results and the confusion matrix

The 10-fold cross-validation on the training split gives:

- **Mean accuracy:** 96.60 %
- **Standard deviation:** 2.58 %

On the 140-sample held-out test set, the model scores **97.14 %** accuracy. The confusion matrix (rows are the true class, columns the predicted class; Class 2 = benign, Class 4 = malignant):

|  | Predicted benign | Predicted malignant |
|---|---:|---:|
| **Actually benign** | 82 (TN) | 3 (FP) |
| **Actually malignant** | 1 (FN) | 54 (TP) |

- **True negatives (TN):** 82 benign cases correctly identified.
- **True positives (TP):** 54 malignant cases correctly identified.
- **False positives (FP):** 3 benign cases misclassified as malignant.
- **False negatives (FN):** 1 malignant case misclassified as benign. **This is the most critical error in a medical setting**, because a missed malignancy is far costlier than a false alarm.

### Caveats

- The test set is small (140 samples), so a single misclassification moves test accuracy by about 0.7 points.
- The mean imputer is fitted on the full dataset before the train/test split, so the 16 imputed `Bare_nuclei` values use a mean that includes test rows. Fitting it on the training split only, for example inside a `Pipeline`, would avoid this small leak.
- The model uses the default 0.5 decision threshold. In a screening context, lowering the threshold would trade more false positives for fewer missed malignancies.

---

## How to run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/AryadipMridha/Logistic_Regression-Breast_Cancer_Original.git
    cd Logistic_Regression-Breast_Cancer_Original
    ```

2. **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn ucimlrepo
    ```

4. **Run the script, or open the notebook:**
    ```bash
    python breast_cancer_logistic_regression.py
    # or
    jupyter notebook breast_cancer_logistic_regression.ipynb
    ```

The script prints the missing-value counts, the test-set confusion matrix and accuracy, and the cross-validated accuracy. The dataset is downloaded from UCI on the first run, so an internet connection is required.

## Files

| File | Contents |
|---|---|
| `breast_cancer_logistic_regression.ipynb` | Colab notebook with saved outputs |
| `breast_cancer_logistic_regression.py` | The same pipeline as a plain Python script |
