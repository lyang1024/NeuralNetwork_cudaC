
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = neural_net
OBJ	        = main.o backprop.o

default: $(EXE)

main.o: main.cu nn_kernel.cu backprop.h support.h
	$(NVCC) -c -g -o $@ main.cu $(NVCC_FLAGS)

backprop.o: backprop.cu backprop.h
	$(NVCC) -c -g -o $@ backprop.cu $(NVCC_FLAGS)
support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
