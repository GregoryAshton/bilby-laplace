PYTHON ?= python

laplace rejection smc dynesty:
	$(PYTHON) gaussian_example.py --sampler $@

compare:
	$(PYTHON) gaussian_example.py --compare

all: laplace rejection smc dynesty compare

.PHONY: all compare laplace rejection smc dynesty
