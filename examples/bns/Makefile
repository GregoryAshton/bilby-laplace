PYTHON ?= python

std-compare:
	$(PYTHON) bns_heterodyned_example.py --likelihood std --compare

rb-compare:
	$(PYTHON) bns_heterodyned_example.py --likelihood rb --compare

mb-compare:
	$(PYTHON) bns_heterodyned_example.py --likelihood mb --compare

compare:
	$(PYTHON) bns_heterodyned_example.py --compare

std-%:
	$(PYTHON) bns_heterodyned_example.py --likelihood std --sampler $*

rb-%:
	$(PYTHON) bns_heterodyned_example.py --likelihood rb --sampler $*

mb-%:
	$(PYTHON) bns_heterodyned_example.py --likelihood mb --sampler $*

std-all: std-laplace std-rejection std-smc std-dynesty std-compare
rb-all: rb-laplace rb-rejection rb-smc rb-dynesty rb-compare
mb-all: mb-laplace mb-rejection mb-smc mb-dynesty mb-compare
all: std-all rb-all mb-all compare

.PHONY: all compare std-all rb-all mb-all std-compare rb-compare mb-compare
