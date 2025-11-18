# Makefile for STATS 604 Project 4

PYTHON := python3

.PHONY: all notebook train predictions clean

# Default: run notebook + training
all: notebook train

# Execute main.ipynb in-place (EDA / analysis)
notebook: main.ipynb
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace main.ipynb

# Heavy training: fit GP and save artifacts (e.g. data/artifacts/gp_artifacts.joblib)
train: train.py
	$(PYTHON) train.py

# Light prediction: load artifacts and print predictions to stdout
# You can redirect if you want: make predictions > predictions.csv
predictions: predict.py
	$(PYTHON) predict.py

# Optional cleanup
clean:
	rm -f predictions.csv
	rm -rf data/artifacts
