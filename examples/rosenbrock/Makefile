PYTHON ?= python

laplace rejection smc emcee dynesty:
	$(PYTHON) run.py --sampler $@

aspire:
	$(PYTHON) run.py --sampler smc-direct

compare:
	$(PYTHON) run.py --compare

all: laplace rejection smc aspire emcee dynesty compare

.PHONY: all compare laplace rejection smc aspire emcee dynesty
