CC := gcc
SRCEXT := c
HEADEREXT := h
CFLAGS := -Wall -Werror -Wmissing-prototypes -std=c11 -O2

NVCC := /usr/local/cuda/bin/nvcc
CUDAFLAGS :=
CUDALIBPATH := /usr/local/cuda/lib64
NVSRCEXT := cu
NVHEADEREXT := h

TARGET = ConjugateGradient
SRCDIR := src
BUILDDIR := build

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
LDFLAGS +=  -liomp5 -lpthread -lm -ldl

# add cuda flags
# -DMKL_ILP64 sets int to 64, has to be added to both gcc and nvcc
CUDAFLAGS := -L$(CUDALIBPATH) -lcuda -lcudart -lcublas -lcusparse -m64 -DMKL_ILP64

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	$(CC) $^ -o $(TARGET) $(LFLAGS) $(LDFLAGS) $(CUDAFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(NVSRCEXT)
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

valgrind:
	valgrind --leak-check=yes ./$(TARGET)

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

cuda_info:
	nvidia-smi -a

.PHONY: default all clean