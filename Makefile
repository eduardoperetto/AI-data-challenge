PYTHON := python3
VENV := venv

# Default target
all: run

# Create a virtual environment
$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

# Install dependencies
install: $(VENV)/bin/activate

# KNN
prepare_knn: install
	ALG := knn
	$(VENV)/bin/python ./$(ALG)/prepare.py

learn_knn: install
	ALG := knn
	$(VENV)/bin/python ./$(ALG)/learn.py

eval_knn: install
	ALG := knn
	$(VENV)/bin/python ./$(ALG)/eval.py

# Clean up the virtual environment
clean:
	rm -rf $(VENV)

.PHONY: all install run clean