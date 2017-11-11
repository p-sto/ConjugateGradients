CC := gcc
CFLAGS := -Wall -Werror -Wmissing-prototypes -std=c11 -O2
TARGET = ConjugateGradient
SRCEXT := c
HEADEREXT := h
SRCDIR := src
BUILDDIR := build
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
HEADERS = $(wildcard $(SRCDIR)/*.$(HEADEREXT))
INC := -I include

# MKL setting
MKLROOT := /opt/intel/mkl
INCLUDE_DIRS = $(MKLROOT)/include
LIBRARY_DIRS := $(MKLROOT)/lib/intel64

LFLAGS := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
# static liking to MKL
LDFLAGS := -Wl,--start-group $(LIBRARY_DIRS)/libmkl_intel_ilp64.a $(LIBRARY_DIRS)/libmkl_intel_thread.a $(LIBRARY_DIRS)/libmkl_core.a -Wl,--end-group
LDFLAGS += -liomp5 -lpthread -lm -ldl -fPIC

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	$(CC) $^ -o $(TARGET) $(LFLAGS) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning...";
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"; $(RM) -r $(BUILDDIR) $(TARGET)

.PHONY: default all clean