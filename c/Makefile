CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99
LDFLAGS ?= -lm

COSMOCC ?= $(shell command -v cosmocc 2>/dev/null || echo "/home/phil/.gemini/antigravity/scratch/cosmocc/bin/cosmocc")
COSMO_CFLAGS ?= -O3 -Wall -Wextra -std=c99
COSMO_LDFLAGS ?=
TMPDIR ?= /tmp

BIN_DIR := bin
SRC := prime_moment.c prime_gtrm.c prime_buckingham.c prime_server.c prime_weights.c prime_net.c prime_gemm_ternary.c prime_tier.c prime_gemm_bipolar.c prime_fractal_memory.c prime_lie_su2.c main.c
HEADERS := prime_moment.h prime_gtrm.h prime_buckingham.h prime_server.h prime_weights.h prime_net.h prime_gemm_ternary.h prime_tier.h prime_gemm_bipolar.h prime_fractal_memory.h prime_lie_su2.h

.PHONY: all native cosmo shared zipalign test test-bundle bench clean help

all: native cosmo zipalign

native: $(BIN_DIR)/prime

$(BIN_DIR)/prime: $(SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)
	@echo "Built native binary: $@"

zipalign: $(BIN_DIR)/zipalign

$(BIN_DIR)/zipalign: zipalign.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ zipalign.c
	@echo "Built zipalign tool: $@"

shared: $(BIN_DIR)/libprime.so

$(BIN_DIR)/libprime.so: prime_moment.c $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ prime_moment.c $(LDFLAGS)
	@echo "Built shared library: $@"

cosmo: $(BIN_DIR)/prime.com

$(BIN_DIR)/prime.com: $(SRC) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	@mkdir -p /home/phil/.gemini/antigravity/scratch/tmp
	HOME=/home/phil/.gemini/antigravity/scratch TMPDIR=/home/phil/.gemini/antigravity/scratch/tmp $(COSMOCC) $(COSMO_CFLAGS) -o $@ $(SRC) $(COSMO_LDFLAGS)
	@echo "Built Cosmopolitan APE binary: $@"

test: native
	$(BIN_DIR)/prime --verify

test-bundle: native zipalign
	@python3 -c "import struct; f=open('bin/dummy_weights.bin','wb'); f.write(struct.pack('4f', 0.1234, 0.5678, 0.9012, 0.3456)); f.close()"
	@cp $(BIN_DIR)/prime $(BIN_DIR)/prime_bundled
	$(BIN_DIR)/zipalign -a 4096 $(BIN_DIR)/prime_bundled bin/dummy_weights.bin weights.bin
	$(BIN_DIR)/prime_bundled --test-mmap weights.bin

bench: native
	$(BIN_DIR)/prime --heads 8 --dim 64 --tokens 10000

bench-limits: $(BIN_DIR)/bench_context_limits
	$(BIN_DIR)/bench_context_limits

$(BIN_DIR)/bench_context_limits: bench_context_limits.c prime_moment.c prime_moment.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ bench_context_limits.c prime_moment.c $(LDFLAGS)
	@echo "Built context limits benchmark: $@"

prove-limits: $(BIN_DIR)/prove_limits
	$(BIN_DIR)/prove_limits

$(BIN_DIR)/prove_limits: prove_limits.c prime_moment.c prime_moment.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -Wl,--wrap=malloc -o $@ prove_limits.c prime_moment.c $(LDFLAGS)
	@echo "Built limits proof suite: $@"

clean:
	rm -rf $(BIN_DIR) *.o

help:
	@echo "Available targets:"
	@echo "  make native  - Build native binary using host GCC"
	@echo "  make cosmo   - Build portable .com binary using Cosmopolitan cosmocc"
	@echo "  make test    - Run self-verification tests"
	@echo "  make bench   - Run benchmark with 10k tokens"
	@echo "  make clean   - Remove build artifacts"
