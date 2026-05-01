PYTHON ?= python

laplace rejection smc dynesty:
	$(PYTHON) run.py --sampler $@

compare:
	$(PYTHON) run.py --compare

all: laplace rejection smc dynesty compare

.PHONY: all compare laplace rejection smc dynesty
