NVCC = nvcc
TARGET = radix_test

SRC = src/main.cu src/histogram.cu src/scan.cu src/scatter.cu src/radix_sort.cu test/cpu_reference.cpp

all:
	$(NVCC) -O3 -std=c++17 -o $(TARGET) $(SRC)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
