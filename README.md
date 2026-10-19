# Chars74K Linear Classification

Binary and multiclass linear classifiers on character images (CSI 5341, University of Ottawa, Fall 2023).

## The Assignment

Using the [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) English image set, working with four characters — **'4', 'A', 'u', 'H'** (Sample005/011/057/009) — converted to grayscale and resized to a common size, the assignment asked to:

- **Binary classification**: train a '4'-vs-rest linear classifier (SGDClassifier / LogisticRegression); evaluate with confusion matrix, accuracy, precision, recall, and ROC curves.
- **Multiclass (One-vs-One)**: compare LogisticRegression (newton-cg, 10→100 iterations), LinearSVC, and SGDClassifier variants on all four characters.
- **Multiclass (One-vs-Rest)**: compare training on the raw imbalanced data vs. SMOTE-balanced data.

## Key Findings

- **Binary '4' detector is solid**: 96% precision on train, 98% on test, with recall/F1 even higher on test than train. ROC curves for train and test follow the same trend but sit away from the ideal (0,1) corner — decent, not perfect.
- **For OvO, the solver matters more than the strategy**: LogisticRegression reaches 85.7% accuracy, but LinearSVC jumps to 95.7% (97.4% precision, 92.2% recall, 94.8% F1). SGD variants with different hyperparameters all land below 80% — linear models are not interchangeable here.
- **'A' is the easiest to find and the hardest to trust**: ~99% recall but the lowest precision (86%) — the model over-predicts 'A', and '4'/'u' are frequently misclassified as 'A'. Classic class-imbalance symptom, which motivated the SMOTE experiment.
- **SMOTE helps the OvR model**: ~90% accuracy on train, ~80% on test after balancing (vs. worse on raw data). 'A' remains the hardest character, most confused with 'H' — visually similar strokes, and no linear boundary separates them cleanly.

## Dataset

The images are not bundled here. Download `EnglishImg.tgz` from the [Chars74K download page](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) (University of Surrey) and extract it so the layout is:

```
English/Img/GoodImg/Bmp/    # 62 subdirectories, one per character/digit
```

Place it next to the notebook to run it (only the '4', 'A', 'u', 'H' folders are used).

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook chars74k_linear_classification.ipynb
```

## Repository Structure

```
├── chars74k_linear_classification.ipynb  # binary, OvO, OvR + SMOTE experiments
└── requirements.txt
```
