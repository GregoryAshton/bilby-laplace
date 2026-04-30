PYTHON ?= python

laplace rejection smc dynesty:
	$(PYTHON) pulse_example.py --sampler $@

compare:
	$(PYTHON) pulse_example.py --compare

all: laplace rejection smc dynesty compare

.PHONY: all compare laplace rejection smc dynesty
