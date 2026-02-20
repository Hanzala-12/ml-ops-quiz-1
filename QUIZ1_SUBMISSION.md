# Practical Quiz 1: Makefile Automation & GitHub

Repository: `git@github.com:Hanzala-12/ml-ops-quiz-1.git`

## Section A: Makefile Automation

### 1) Project Setup
Created the required ML project structure with a CSV dataset and Logistic Regression pipeline.

```text
data/
  raw/breast_cancer.csv
  processed/
src/
  preprocess.py
  train.py
  evaluate.py
models/
results/
Makefile
requirements.txt
```

Files:
- `src/preprocess.py`: loads CSV, handles missing values, train/test split, saves processed files.
- `src/train.py`: trains Logistic Regression, saves model, prints accuracy.
- `src/evaluate.py`: evaluates saved model and writes final report.

### 2) Data Preprocessing Automation
The preprocessing script reads `data/raw/breast_cancer.csv`, fills missing numeric values with median, and splits data into train/test with stratification.  
Processed files are saved to `data/processed/` as `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.  
This is automated through the `make preprocess` target in `Makefile`.

Command:
```powershell
make preprocess
```

Observed output:
```text
python src/preprocess.py
Loaded dataset shape: (569, 31)
Missing values before fill: 841
Training samples: 455
Testing samples: 114
Processed files saved to: data\processed
```

### 3) Model Training Automation
Training uses `LogisticRegression` from scikit-learn on processed data.  
The trained model is saved at `models/logistic_regression.joblib`.  
Accuracy is printed to terminal and also saved in `results/train_accuracy.txt`.

Command:
```powershell
make train
```

Observed output:
```text
python src/train.py
Model saved to: models\logistic_regression.joblib
Training completed. Accuracy: 0.9474
Training classification report saved to: results\train_report.txt
```

### 4) Results Generation Pipeline (`make all`)
The `all` target runs the complete pipeline in sequence: `preprocess -> train -> evaluate`.  
This provides one command for full automation and repeatability.  
Final evaluation metrics are written to `results/evaluation.txt`.

Command:
```powershell
make all
```

Observed output:
```text
python src/preprocess.py
...
python src/train.py
Training completed. Accuracy: 0.9474
python src/evaluate.py
Evaluation completed. Accuracy: 0.9474
Detailed report saved to: results\evaluation.txt
```

Final accuracy: **0.9474**

## Section B: GitHub Practical & Concepts

### 1) Basic Git Commands (One-line Explanation)
- `git init`: Initializes a new local Git repository.
- `git add .`: Stages all current file changes for commit.
- `git commit -m "Initial commit"`: Saves staged changes as a commit with a message.
- `git remote add origin <url>`: Links local repo to a remote GitHub repository.
- `git push`: Uploads local commits to the configured remote branch.

### 2) Branching & Collaboration

1. Create branch:
```powershell
git checkout -b feature-ml
```

2. Modify training script: updated `src/train.py` to save a training classification report.

3. Commit and push branch:
```powershell
git add src/train.py
git commit -m "feat: add training classification report output"
git push -u origin feature-ml
```

4. Merge branch into main:
```powershell
git checkout main
git merge --no-ff feature-ml -m "merge: feature-ml into main"
git push
```

Branch list:
```text
feature-ml
* main
remotes/origin/feature-ml
remotes/origin/main
```

Merge history:
```text
*   0613ab3 (HEAD -> main, origin/main) merge: feature-ml into main
|\  
| * bcb593c (origin/feature-ml, feature-ml) feat: add training classification report output
|/  
* 346f0d5 feat: add ml pipeline automation with Makefile
```

### 3) Concept Check (Answers)

1. **Difference between `git pull` and `git fetch`:**  
`git fetch` only downloads new remote commits; `git pull` downloads and also merges (or rebases) into your current branch.

2. **What happens if two people push to the same branch?**  
If one push arrives first, the second person usually gets a non-fast-forward rejection and must pull/rebase/merge before pushing again.

3. **Difference between fork and clone:**  
A fork creates a server-side copy of someone else's repo under your GitHub account; a clone creates a local copy of a remote repo on your machine.

4. **Why use `.gitignore` in ML projects?**  
It prevents large/generated files (models, logs, processed data, cache) from being committed, keeping the repo clean, fast, and reproducible.

## Screenshot Checklist

- Terminal showing `make preprocess` output.
- Terminal showing `make train` output with accuracy.
- Terminal showing `make all` output and final accuracy.
- GitHub repo page showing commits on `main`.
- GitHub repo page showing `feature-ml` branch.
- Terminal showing branch list (`git branch -a`) and merge history (`git log --oneline --graph --decorate --all`).
- Final model result screenshot (`results/evaluation.txt` accuracy/report).
