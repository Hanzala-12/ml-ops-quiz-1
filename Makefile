PYTHON := python

.PHONY: install preprocess train evaluate all clean

install:
	$(PYTHON) -m pip install -r requirements.txt

preprocess:
	$(PYTHON) src/preprocess.py

train:
	$(PYTHON) src/train.py

evaluate:
	$(PYTHON) src/evaluate.py

all: preprocess train evaluate

clean:
	$(PYTHON) -c "from pathlib import Path; [p.unlink(missing_ok=True) for p in Path('models').glob('*.joblib')]; [p.unlink(missing_ok=True) for p in Path('results').glob('*.txt')]; [p.unlink(missing_ok=True) for p in Path('data/processed').glob('*.csv')]"
