CC := gcc
SRCEXT := c
HEADEREXT := h

DEBUG_FLAGS := -g -DDEBUG
CFLAGS := -Wall -Wmissing-prototypes -Wno-unused-result -std=c11 -O2 $(DEBUG_FLAGS) -fopenmp

NVCC := /usr/local/cuda/bin/nvcc
CUDAFLAGS :=
CUDALIBPATH := /usr/local/cuda/lib64
NVSRCEXT := cu
NVHEADEREXT := h

TARGET = ConjugateGradient
TARGET_DEBUG = debug_ConjugateGradient
SRCDIR := src
BUILDDIR := build
BUILDDIR_DEBUG = build_debug

SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
HEADERS = $(wildcard $(SRCDIR)/*.$(HEADEREXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

CUDASOURCES := $(shell find $(SRCDIR) -type f -name *.$(NVSRCEXT))
CUDAHEADERS := $(wildcard $(SRCDIR)/*.$(NVHEADEREXT))
CUDAOBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CUDASOURCES:.$(NVSRCEXT)=.o))

SOURCES += $(CUDASOURCES)
HEADERS += $(CUDAHEADERS)
OBJECTS += $(CUDAOBJECTS)

INC := -I include -I/usr/local/cuda/include

# MKL setting
MKLROOT := /opt/intel/mkl
INCLUDE_DIRS = $(MKLROOT)/include
LIBRARY_DIRS := $(MKLROOT)/lib/intel64

LFLAGS := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CFLAGS += -DMKL_ILP64 -m64 -I${MKLROOT}/include

# static liking to MKL
LDFLAGS := -Wl,--start-group $(LIBRARY_DIRS)/libmkl_intel_ilp64.a $(LIBRARY_DIRS)/libmkl_intel_thread.a $(LIBRARY_DIRS)/libmkl_core.a -Wl,--end-group
LDFLAGS +=  -liomp5 -lpthread -lm -ldl -lstdc++

# add cuda flags
# -DMKL_ILP64 sets int to 64, has to be added to both gcc and nvcc
CUDAFLAGS := -L$(CUDALIBPATH) -lcuda -lcudart -lcublas -lcusparse -m64 -DMKL_ILP64 $(DEBUG_FLAGS)

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	$(CC) $^ -o $(TARGET) $(LFLAGS) $(LDFLAGS) $(CUDAFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(NVSRCEXT)
	@mkdir -p $(@D)
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

valgrind:
	valgrind --leak-check=yes ./$(TARGET)

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)
	@echo " $(RM) -r $(BUILDDIR_DEBUG) $(TARGET_DEBUG)"; $(RM) -r $(BUILDDIR_DEBUG) $(TARGET_DEBUG)

cuda_info:
	nvidia-smi -a

.PHONY: default all clean debug cuda_info