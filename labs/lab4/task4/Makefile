COMPILER=nvcc
LIBS=-lm

all: task4

%: %.cu
	$(COMPILER) $^ -o $@ $(LIBS)

clean:
	rm -rf task4
