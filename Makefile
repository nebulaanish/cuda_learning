# Variables
NVCC = nvcc
CCBIN = $(shell which x86_64-conda-linux-gnu-g++)
ARCH = sm_90
TARGET = builds/histogram
SRC = code/007_histogram.cu

# Compilation command
all:
	$(NVCC) -ccbin $(CCBIN) -arch=$(ARCH) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)