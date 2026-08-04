PYTHON ?= python

laplace rejection smc smc-direct dynesty:
	$(PYTHON) run.py --sampler $@

compare:
	$(PYTHON) run.py --compare

all: laplace rejection smc smc-direct dynesty compare

.PHONY: all compare laplace rejection smc smc-direct dynesty
