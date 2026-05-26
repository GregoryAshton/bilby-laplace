PYTHON ?= python

laplace rejection rejection-user smc dynesty:
	$(PYTHON) run.py --sampler $@

compare:
	$(PYTHON) run.py --compare

all: laplace rejection rejection-user smc dynesty compare

.PHONY: all compare laplace rejection rejection-user smc dynesty
