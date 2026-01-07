# Variables
NVCC = nvcc
CCBIN = $(shell which x86_64-conda-linux-gnu-g++)
ARCH = sm_61
TARGET = vector_add
SRC = vector_add.cu

# Compilation command
all:
	$(NVCC) -ccbin $(CCBIN) -arch=$(ARCH) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)