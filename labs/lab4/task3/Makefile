COMPILER=nvcc
LIBS=-lm

all: task3

%: %.cu
	$(COMPILER) $^ -o $@ $(LIBS)

clean:
	rm -rf task3
