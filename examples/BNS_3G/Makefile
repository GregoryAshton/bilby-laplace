PYTHON ?= python

std-compare:
	$(PYTHON) run.py --likelihood std --compare

rb-compare:
	$(PYTHON) run.py --likelihood rb --compare

compare:
	$(PYTHON) run.py --compare

std-%:
	$(PYTHON) run.py --likelihood std --sampler $*

rb-%:
	$(PYTHON) run.py --likelihood rb --sampler $*

std-all: std-laplace std-rejection std-smc std-smc-direct std-dynesty std-compare
rb-all: rb-laplace rb-rejection rb-smc rb-smc-direct rb-dynesty rb-compare
all: std-all rb-all compare

.PHONY: all compare std-all rb-all std-compare rb-compare validate

validate:
	$(PYTHON) run.py --validate-likelihood
