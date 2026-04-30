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
# Pulse example
# ---------------------------------------------------------------------------
pulse-%:
	$(PYTHON) pulse_example.py --sampler $*

pulse-all: pulse-laplace pulse-rejection pulse-smc pulse-dynesty pulse-compare

pulse-compare:
	$(PYTHON) pulse_example.py --compare

# ---------------------------------------------------------------------------
# BNS example — standard (std), relative binning (rb), and multi-banding (mb)
# ---------------------------------------------------------------------------
bns-std-%:
	$(PYTHON) bns_heterodyned_example.py --likelihood std --sampler $*

bns-std-all: bns-std-laplace bns-std-rejection bns-std-smc bns-std-dynesty bns-std-compare

bns-std-compare:
	$(PYTHON) bns_heterodyned_example.py --likelihood std --compare

bns-rb-%:
	$(PYTHON) bns_heterodyned_example.py --likelihood rb --sampler $*

bns-rb-all: bns-rb-laplace bns-rb-rejection bns-rb-smc bns-rb-dynesty bns-rb-compare

bns-rb-compare:
	$(PYTHON) bns_heterodyned_example.py --likelihood rb --compare

bns-mb-%:
	$(PYTHON) bns_heterodyned_example.py --likelihood mb --sampler $*

bns-mb-all: bns-mb-laplace bns-mb-rejection bns-mb-smc bns-mb-dynesty bns-mb-compare

bns-mb-compare:
	$(PYTHON) bns_heterodyned_example.py --likelihood mb --compare

bns-compare:
	$(PYTHON) bns_heterodyned_example.py --compare

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help gaussian-all gaussian-compare rosenbrock-all rosenbrock-compare lorentzian-all lorentzian-compare hlv-all hlv-compare pulse-all pulse-compare bns-std-all bns-std-compare bns-rb-all bns-rb-compare bns-mb-all bns-mb-compare bns-compare

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
	@echo "Pulse example (time-series Gaussian pulse):"
	@echo "  pulse-laplace         Run Laplace approximation (no resampling)"
	@echo "  pulse-rejection       Run Laplace with rejection resampling"
	@echo "  pulse-smc             Run Laplace with SMC resampling"
	@echo "  pulse-dynesty         Run dynesty nested sampling"
	@echo "  pulse-compare         Compare all existing pulse results"
	@echo "  pulse-all             Run all pulse samplers then compare"
	@echo ""
	@echo "BNS example — standard likelihood (ET + CE):"
	@echo "  bns-std-laplace    Run Laplace (standard)"
	@echo "  bns-std-rejection  Run rejection resampling (standard)"
	@echo "  bns-std-smc        Run SMC resampling (standard)"
	@echo "  bns-std-dynesty    Run dynesty (standard)"
	@echo "  bns-std-compare    Compare existing std results"
	@echo "  bns-std-all        Run all std samplers then compare"
	@echo ""
	@echo "BNS example — relative binning (ET + CE):"
	@echo "  bns-rb-laplace    Run Laplace (relative binning)"
	@echo "  bns-rb-rejection  Run rejection resampling (relative binning)"
	@echo "  bns-rb-smc        Run SMC resampling (relative binning)"
	@echo "  bns-rb-dynesty    Run dynesty (relative binning)"
	@echo "  bns-rb-compare    Compare existing rb results"
	@echo "  bns-rb-all        Run all rb samplers then compare"
	@echo ""
	@echo "BNS example — multi-banding (ET + CE):"
	@echo "  bns-mb-laplace    Run Laplace (multi-banding)"
	@echo "  bns-mb-rejection  Run rejection resampling (multi-banding)"
	@echo "  bns-mb-smc        Run SMC resampling (multi-banding)"
	@echo "  bns-mb-dynesty    Run dynesty (multi-banding)"
	@echo "  bns-mb-compare    Compare existing mb results"
	@echo "  bns-mb-all        Run all mb samplers then compare"
	@echo ""
	@echo "  bns-compare       Compare ALL BNS results across all likelihoods"
