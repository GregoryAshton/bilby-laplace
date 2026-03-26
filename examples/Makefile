PYTHON ?= python

# ---------------------------------------------------------------------------
# Gaussian example
# ---------------------------------------------------------------------------
gaussian-%:
	$(PYTHON) gaussian_example.py --sampler $*

gaussian-all: gaussian-laplace gaussian-rejection gaussian-smc gaussian-dynesty gaussian-compare

gaussian-compare:
	$(PYTHON) gaussian_example.py --compare

# ---------------------------------------------------------------------------
# Rosenbrock example
# ---------------------------------------------------------------------------
rosenbrock-%:
	$(PYTHON) rosenbrock_example.py --sampler $*

rosenbrock-all: rosenbrock-laplace rosenbrock-rejection rosenbrock-smc rosenbrock-dynesty rosenbrock-compare

rosenbrock-compare:
	$(PYTHON) rosenbrock_example.py --compare

# ---------------------------------------------------------------------------
# Lorentzian example
# ---------------------------------------------------------------------------
lorentzian-%:
	$(PYTHON) lorentzian_example.py --sampler $*

lorentzian-all: lorentzian-laplace lorentzian-rejection lorentzian-smc lorentzian-dynesty lorentzian-compare

lorentzian-compare:
	$(PYTHON) lorentzian_example.py --compare

# ---------------------------------------------------------------------------
# HLV injection example
# ---------------------------------------------------------------------------
hlv-%:
	$(PYTHON) hlv_example.py --sampler $*

hlv-all: hlv-laplace hlv-rejection hlv-smc hlv-dynesty hlv-compare

hlv-compare:
	$(PYTHON) hlv_example.py --compare

# ---------------------------------------------------------------------------
# GW150914 example
# ---------------------------------------------------------------------------
GW150914-%:
	$(PYTHON) GW150914.py --sampler $*

GW150914-all: GW150914-laplace GW150914-rejection GW150914-smc GW150914-dynesty GW150914-compare

GW150914-compare:
	$(PYTHON) GW150914.py --compare

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help gaussian-all gaussian-compare rosenbrock-all rosenbrock-compare lorentzian-all lorentzian-compare hlv-all hlv-compare gw150914-all gw150914-compare

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Gaussian example:"
	@echo "  gaussian-laplace      Run Laplace approximation (no resampling)"
	@echo "  gaussian-rejection    Run Laplace with rejection resampling"
	@echo "  gaussian-smc          Run Laplace with SMC resampling"
	@echo "  gaussian-dynesty      Run dynesty nested sampling"
	@echo "  gaussian-compare      Compare all existing Gaussian results"
	@echo "  gaussian-all          Run all Gaussian samplers then compare"
	@echo ""
	@echo "Rosenbrock example:"
	@echo "  rosenbrock-laplace    Run Laplace approximation (no resampling)"
	@echo "  rosenbrock-rejection  Run Laplace with rejection resampling"
	@echo "  rosenbrock-smc        Run Laplace with SMC resampling"
	@echo "  rosenbrock-dynesty    Run dynesty nested sampling"
	@echo "  rosenbrock-compare    Compare all existing Rosenbrock results"
	@echo "  rosenbrock-all        Run all Rosenbrock samplers then compare"
	@echo ""
	@echo "Lorentzian example (high-SNR non-Gaussian):"
	@echo "  lorentzian-laplace    Run Laplace approximation (no resampling)"
	@echo "  lorentzian-rejection  Run Laplace with rejection resampling"
	@echo "  lorentzian-smc        Run Laplace with SMC resampling"
	@echo "  lorentzian-dynesty    Run dynesty nested sampling"
	@echo "  lorentzian-compare    Compare all existing Lorentzian results"
	@echo "  lorentzian-all        Run all Lorentzian samplers then compare"
	@echo ""
	@echo "HLV injection example:"
	@echo "  hlv-laplace           Run Laplace approximation (no resampling)"
	@echo "  hlv-rejection         Run Laplace with rejection resampling"
	@echo "  hlv-smc               Run Laplace with SMC resampling"
	@echo "  hlv-dynesty           Run dynesty nested sampling"
	@echo "  hlv-compare           Compare all existing HLV results"
	@echo "  hlv-all               Run all HLV samplers then compare"
	@echo ""
	@echo "GW150914 example:"
	@echo "  gw150914-laplace      Run Laplace approximation (no resampling)"
	@echo "  gw150914-rejection    Run Laplace with rejection resampling"
	@echo "  gw150914-smc          Run Laplace with SMC resampling"
	@echo "  gw150914-dynesty      Run dynesty nested sampling"
	@echo "  gw150914-compare      Compare all existing GW150914 results"
	@echo "  gw150914-all          Run all GW150914 samplers then compare"
