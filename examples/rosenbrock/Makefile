PYTHON ?= python

laplace rejection smc dynesty:
	$(PYTHON) run.py --sampler $@

aspire:
	$(PYTHON) run.py --sampler smc-direct

compare:
	$(PYTHON) run.py --compare

all: laplace rejection smc aspire dynesty compare

.PHONY: all compare laplace rejection smc aspire dynesty
