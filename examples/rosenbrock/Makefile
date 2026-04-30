PYTHON ?= python

laplace rejection smc dynesty:
	$(PYTHON) rosenbrock_example.py --sampler $@

compare:
	$(PYTHON) rosenbrock_example.py --compare

all: laplace rejection smc dynesty compare

.PHONY: all compare laplace rejection smc dynesty
