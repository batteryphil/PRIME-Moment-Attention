# Root Makefile for PRIME Moment Attention
# Provides zero-friction targets for low-level systems engineers

.PHONY: all native cosmo test test-bundle bench clean help

all:
	@$(MAKE) -C c all

native:
	@$(MAKE) -C c native

cosmo:
	@mkdir -p tmp
	@TMPDIR=$$(pwd)/tmp HOME=$$(pwd)/tmp $(MAKE) -C c cosmo

test:
	@$(MAKE) -C c test

test-bundle:
	@$(MAKE) -C c test-bundle

bench:
	@$(MAKE) -C c bench

clean:
	@$(MAKE) -C c clean
	@rm -rf tmp

help:
	@echo "PRIME Moment Attention - Zero-Dependency C99 & Cosmopolitan Engine"
	@echo "Targets:"
	@echo "  make bench        - Compile and run 10,000-token decode latency & throughput benchmark"
	@echo "  make native       - Build native C99 binary (c/bin/prime)"
	@echo "  make test         - Run numerical stability & zero-copy mmap tests"
	@echo "  make cosmo        - Build Actually Portable Executable (c/bin/prime.com) via Cosmopolitan"
	@echo "  make test-bundle  - Test page-aligned zip weight bundling & zero-copy mmap"
	@echo "  make clean        - Remove build artifacts"
