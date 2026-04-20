NVCC = nvcc
TARGET = radix_test

SRC = src/main.cu src/histogram.cu

all:
	$(NVCC) -O3 -o $(TARGET) $(SRC)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
