NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_86

TARGET = radix_test
SRC = src/main.cu src/radix_sort.cu src/histogram.cu src/scan.cu src/scatter.cu

all:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)