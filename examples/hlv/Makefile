PYTHON ?= python

laplace rejection smc dynesty:
	$(PYTHON) hlv_example.py --sampler $@

compare:
	$(PYTHON) hlv_example.py --compare

all: laplace rejection smc dynesty compare

.PHONY: all compare laplace rejection smc dynesty
