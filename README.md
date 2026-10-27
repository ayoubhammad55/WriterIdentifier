# Handwritten Writer Identification

A machine-learning project for identifying the writer of a handwritten document using image-processing techniques, classical machine learning, and deep learning.

The project evaluates different approaches on the IAM Handwriting Database and compares their performance using a consistent experimental setup.

## Objectives

- Develop a writer identification pipeline for handwritten document images.
- Evaluate Local Binary Patterns (LBP) combined with Support Vector Machines (SVM).
- Train and compare grayscale ResNet-18 models, including a dropout variant.
- Measure performance using test accuracy and macro F1-score.
- Document experimental results and provide a reproducible workflow for further research.

## Methods

### LBP + SVM

Uses Local Binary Patterns to extract texture features from handwriting images and a Support Vector Machine to classify writers.

### ResNet-18

Uses a convolutional neural network to learn visual representations from grayscale handwriting images.

### ResNet-18 with Dropout

Evaluates a dropout variant of ResNet-18 to investigate its effect on generalization.

## Experimental Results

The three approaches were evaluated using the same data split, with 93 writer classes and random seed 42.

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| LBP + SVM | 97.35% | 95.88% |
| ResNet-18 with Dropout | 80.53% | 74.70% |
| ResNet-18 | 80.53% | 73.49% |

The LBP + SVM approach achieved the strongest performance in these experiments. The results suggest that handcrafted texture features can be highly effective for this particular writer identification setup.

These results should be interpreted within the evaluation protocol described below.

## Evaluation Protocol

- **Number of writer classes:** 93
- **Training images:** 441
- **Validation images:** 113
- **Test images:** 113
- **Random seed:** 42
- **Task:** Closed-set writer identification

The evaluation uses a closed-set setup: writers represented in the test set are also represented in training. The experiment separates forms across the splits, but it does not establish writer-independent generalization to previously unseen identities.

## Repository Structure

```text
WriterIdentifier/
├── notebooks/
│   └── iam_writer_identification_reproducible.ipynb
├── results/
│   ├── experiment_audit.json
│   ├── iam_form_level_audit.json
│   ├── model_comparison_same_split.csv
│   └── model_comparison_same_split.png
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pre_processing/
│   ├── segmentation/
│   ├── utils/
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Python 3.10 or 3.11 is a reasonable starting point, but the environment has not been verified across all Python versions.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Experiments

Open the notebook:

```text
notebooks/iam_writer_identification_reproducible.ipynb
```

The notebook currently contains Kaggle-specific dataset paths. Update those paths to match your environment and make sure you have access to the required datasets before running it.

## Dataset

The experiments use the IAM Handwriting Database.

Dataset information: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

The dataset is not included in this repository. Users must obtain access and follow the dataset's terms of use.

## Limitations and Future Work

- Evaluate performance on more handwriting samples per writer.
- Investigate more robust preprocessing and data augmentation.
- Compare additional CNN architectures and feature extraction methods.
- Evaluate generalization across documents and writing conditions.
- Improve reproducibility by documenting exact dependency versions and dataset preparation.

## Technologies

Python, OpenCV, NumPy, pandas, scikit-image, scikit-learn, PyTorch, Torchvision, Matplotlib, and joblib.

## License

A license has not yet been specified for this repository. Add an appropriate license before distributing the code for reuse.
