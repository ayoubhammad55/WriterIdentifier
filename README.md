
# Handwritten Writer Identification

A machine-learning project for identifying the writer of a handwritten document using image-processing techniques, classical machine learning, and deep learning.

This project evaluates Local Binary Patterns (LBP) combined with a Support Vector Machine (SVM) and two grayscale ResNet-18 variants on the IAM Handwriting Database. The experiments use a consistent data split to compare the models under the same evaluation conditions.

## Project Status

This repository documents an experimental comparison of classical machine learning and deep learning methods for closed-set handwritten writer identification.

The reported metrics reflect the specific dataset split and experimental setup described below. They should not be interpreted as evidence of generalization to previously unseen writer identities.

## Objectives

- Develop a writer identification pipeline for handwritten document images.
- Evaluate Local Binary Patterns (LBP) combined with Support Vector Machines (SVM).
- Train and compare grayscale ResNet-18 models, including a dropout variant.
- Measure performance using test accuracy and macro F1-score.
- Document experimental results and provide a reproducible workflow for further research.

## Methods

### LBP + SVM

Local Binary Patterns extract handcrafted texture features from handwriting images. A Support Vector Machine uses these features to classify the writer.

### ResNet-18

A convolutional neural network learns visual representations from grayscale handwriting images and predicts the writer class.

### ResNet-18 with Dropout

This variant incorporates dropout to investigate its effect on model generalization and classification performance.

## Experimental Results

The three approaches were evaluated using the same data split, with 93 writer classes and random seed 42.

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| LBP + SVM | 97.35% | 95.88% |
| ResNet-18 with Dropout | 80.53% | 74.70% |
| ResNet-18 | 80.53% | 73.49% |

The LBP + SVM approach achieved the strongest performance in these experiments. These results suggest that handcrafted texture features can be highly effective for this particular writer identification setup.

The results apply to the experimental protocol described below and should not be interpreted as proof that LBP + SVM will outperform deep learning approaches on other datasets or evaluation settings.

## Evaluation Protocol

| Setting | Value |
|---|---|
| Dataset | IAM Handwriting Database |
| Number of writer classes | 93 |
| Training images | 441 |
| Validation images | 113 |
| Test images | 113 |
| Random seed | 42 |
| Task | Closed-set writer identification |

### Evaluation limitations

This experiment uses a closed-set evaluation protocol: writers represented in the test set are also represented in the training set.

Forms are separated across the training, validation, and test splits. However, this setup does not establish writer-independent generalization to previously unseen identities.

The dataset split is relatively small, so the reported metrics should be interpreted in the context of the number of samples and writer classes evaluated.

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
│   └── utils/
├── .gitignore
├── README.md
└── requirements.txt
```

The notebook contains the main experimental workflow. The `src/` directory contains supporting data-processing, feature-extraction, model, preprocessing, segmentation, and utility modules used by the project.

## Installation

Python 3.10 or 3.11 is a reasonable starting point, but compatibility has not been verified across all Python versions.

### 1. Clone the repository

```bash
git clone https://github.com/ayoubhammad55/WriterIdentifier.git
cd WriterIdentifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependency compatibility may vary by operating system, Python version, and installed machine-learning libraries.

## Running the Experiments

Open the notebook:

```text
notebooks/iam_writer_identification_reproducible.ipynb
```

The notebook currently contains Kaggle-specific dataset paths. Before running it in another environment, update those paths to match your local dataset location.

Ensure that the required IAM data files are available and that the dataset has been prepared in the format expected by the notebook.

Run the notebook cells in order to reproduce the data preparation, model evaluation, and result-generation workflow.

The reported metrics should be considered reproduced only after the notebook has been executed successfully in the intended environment.

## Dataset

This project uses the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), a database of handwritten English text used for handwriting recognition, writer identification, and writer verification research.

**The IAM dataset is not included in this repository.** Users must obtain access through the official website and comply with the database's terms of use.

According to the IAM website:

- The database may be used for **non-commercial research purposes only**.
- Users must register to access the IAM Handwriting Database 3.0 download.
- Publications based on the database should cite the reference listed below.

Please consult the [official IAM Handwriting Database website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) for the current access conditions and terms of use.

### Required dataset reference

Marti, U.-V., & Bunke, H. (2002). The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. *International Journal of Document Analysis and Recognition*, 5, 39–46.

https://doi.org/10.1007/S100320200071

This repository contains the implementation, experimental code, and results for the writer identification experiments. It does not redistribute the IAM dataset.

## Limitations and Future Work

Potential directions for further investigation include:

- Evaluate performance using more handwriting samples per writer.
- Investigate more robust preprocessing and data augmentation.
- Compare additional CNN architectures and feature-extraction methods.
- Evaluate generalization across documents, writing conditions, and previously unseen writers.
- Investigate writer-independent evaluation protocols.
- Improve reproducibility by documenting exact dependency versions and dataset preparation procedures.

## Technologies

- Python
- OpenCV
- NumPy
- pandas
- scikit-image
- scikit-learn
- PyTorch
- Torchvision
- Matplotlib
- joblib

## License

A license has not yet been specified for this repository.

Before distributing the project's code for reuse, consider adding an appropriate software license. Any code license should be kept separate from the IAM dataset's terms of use. Licensing this repository does not grant permission to redistribute or use the dataset beyond its permitted terms.
