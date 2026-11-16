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
MPICXX   ?= mpicxx

BUILD      := build
COMMON_SRC := $(wildcard src/common/*.cpp)
COMMON_OBJ := $(COMMON_SRC:src/%.cpp=$(BUILD)/obj/%.o)
BINARIES   := $(BUILD)/seq $(BUILD)/omp $(BUILD)/threads $(BUILD)/memexp \
              $(BUILD)/hadoop/mapper_stats $(BUILD)/hadoop/reducer_stats \
              $(BUILD)/hadoop/mapper_normalize $(BUILD)/hadoop_finalize \
              $(BUILD)/rpc_server $(BUILD)/rpc_client
# Bonus: the CUDA version is built only where nvcc is installed.
NVCC := $(firstword $(shell command -v nvcc 2>/dev/null) $(wildcard /opt/cuda/bin/nvcc))
ifneq ($(NVCC),)
BINARIES   += $(BUILD)/cuda
endif
# Phase 2 binaries are built only where an MPI compiler is installed.
ifneq ($(shell command -v $(MPICXX) 2>/dev/null),)
BINARIES   += $(BUILD)/mpi $(BUILD)/pingpong
endif

.PHONY: all data test check report clean
.SECONDARY:
all: $(BINARIES)

$(BUILD)/obj/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/omp/%.o: src/omp/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/obj/memexp/%.o: src/memexp/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/obj/mpi/%.o: src/mpi/%.cpp
	@mkdir -p $(@D)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(BUILD)/mpi: $(BUILD)/obj/mpi/main.o $(COMMON_OBJ)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(BUILD)/pingpong: $(BUILD)/obj/mpi/pingpong.o
	$(MPICXX) $(CXXFLAGS) $^ -o $@

# Phase 3: Hadoop Streaming programs. Statically linked, because they run inside the Hadoop
# containers, whose C/C++ runtime differs from the host's.
$(BUILD)/hadoop/%: $(BUILD)/obj/hadoop/%.o $(COMMON_OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -static $^ -o $@

$(BUILD)/hadoop_finalize: $(BUILD)/obj/hadoop/finalize.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Bonus: GPU version. nvcc compiles the .cu file and links the g++-built common objects.
$(BUILD)/cuda: src/cuda/main.cu $(COMMON_OBJ) src/common/app.hpp src/common/kernels.hpp
	$(NVCC) -O3 -std=c++20 -Xcompiler -fopenmp src/cuda/main.cu $(COMMON_OBJ) -o $@ -lgomp

# Bonus: TCP query service in front of the results.
$(BUILD)/rpc_server: $(BUILD)/obj/rpc/server.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) -pthread $^ -o $@

$(BUILD)/rpc_client: $(BUILD)/obj/rpc/client.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/seq: $(BUILD)/obj/seq/main.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/omp: $(BUILD)/obj/omp/main.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

$(BUILD)/threads: $(BUILD)/obj/threads/main.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) -pthread $^ -o $@

$(BUILD)/memexp: $(BUILD)/obj/memexp/main.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)

# ---- preprocessing (Python, conda env `sw401`) --------------------------------------------
data:
	$(PYTHON) preprocessing/validate.py
	$(PYTHON) preprocessing/predecode.py
	$(PYTHON) preprocessing/make_subsets.py
	$(PYTHON) preprocessing/reference.py
	$(PYTHON) preprocessing/make_subsets.py --sizes 700 1400 2800 5600 --reports results/preprocessing/weak
	$(PYTHON) preprocessing/reference.py --sets 700MB 1400MB 2800MB 5600MB
	$(PYTHON) preprocessing/to_text.py

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
	$(BUILD)/threads --set 100MB
	$(PYTHON) tests/check.py --set 100MB --run results/runs/threads/100MB
ifneq ($(shell command -v $(MPICXX) 2>/dev/null),)
	mpirun -n 4 $(BUILD)/mpi --set 100MB
	$(PYTHON) tests/check.py --set 100MB --run results/runs/mpi/100MB
endif
ifneq ($(NVCC),)
	$(BUILD)/cuda --set 100MB
	$(PYTHON) tests/check.py --set 100MB --run results/runs/cuda/100MB
endif
	@if [ -f data/processed/text/100MB.txt ]; then PYTHON=$(PYTHON) scripts/hadoop/local_test.sh 100MB; \
	 else echo "skipping the MapReduce pipe test: run python preprocessing/to_text.py"; fi

# ---- reports (LaTeX, needs latexmk) -> report.pdf (Phase 1), report_phase2.pdf, report_final.pdf ----
report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex phase2.tex final.tex
	cp report/main.pdf report.pdf
	cp report/phase2.pdf report_phase2.pdf
	cp report/final.pdf report_final.pdf

clean:
	rm -rf $(BUILD)
	cd report && latexmk -C main.tex phase2.tex final.tex
