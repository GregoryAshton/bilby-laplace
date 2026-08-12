PYTHON ?= python

laplace rejection rejection-user smc dynesty:
	$(PYTHON) run.py --sampler $@

aspire:
	$(PYTHON) run.py --sampler smc-direct

compare:
	$(PYTHON) run.py --compare

all: laplace rejection rejection-user smc aspire dynesty compare

.PHONY: all compare laplace rejection rejection-user smc aspire dynesty
