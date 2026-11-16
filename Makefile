# Run everything from the repository root. See README.md for the full run order.
#
#   make              build all C++ implementations into build/
#   make data         full preprocessing pipeline (needs the conda env `sw401` active)
#   make test         run every implementation on the 100MB set and check it bit-exactly
#   make check V=omp SET=full   check one existing run against the reference
#   make clean        remove build/

CXX      ?= g++
PYTHON   ?= python
CXXFLAGS ?= -O3 -march=native
# -ffp-contract=off is required: the reference spec forbids fused multiply-add (bit-exactness).
override CXXFLAGS += -std=c++20 -Wall -Wextra -ffp-contract=off -MMD -MP
OMPFLAGS := -fopenmp

BUILD      := build
COMMON_SRC := $(wildcard src/common/*.cpp)
COMMON_OBJ := $(COMMON_SRC:src/%.cpp=$(BUILD)/obj/%.o)
BINARIES   := $(BUILD)/seq $(BUILD)/omp

.PHONY: all data test check clean
all: $(BINARIES)

$(BUILD)/obj/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/omp/%.o: src/omp/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/seq: $(BUILD)/obj/seq/main.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/omp: $(BUILD)/obj/omp/main.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

# ---- preprocessing (Python, conda env `sw401`) --------------------------------------------
data:
	$(PYTHON) preprocessing/validate.py
	$(PYTHON) preprocessing/predecode.py
	$(PYTHON) preprocessing/make_subsets.py
	$(PYTHON) preprocessing/reference.py

# ---- correctness ---------------------------------------------------------------------------
V   ?= seq
SET ?= 100MB
check:
	$(PYTHON) tests/check.py --set $(SET) --run results/runs/$(V)/$(SET)

test: $(BINARIES)
	$(BUILD)/seq --set 100MB
	$(PYTHON) tests/check.py --set 100MB --run results/runs/seq/100MB
	$(BUILD)/omp --set 100MB --schedule dynamic,1
	$(PYTHON) tests/check.py --set 100MB --run results/runs/omp/100MB

clean:
	rm -rf $(BUILD)
