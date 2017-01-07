#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "nn_kernel.cu"
#include "support.h"

int layer_size = 0;

int main(int argc, char** argv) {
	if (argc != 2) {
		fprintf(stderr, "usage: backprop <num of input elements>\n");
		exit(0);
	}
	layer_size = atoi(argv[1]);
	int seed = 7;
	srand(seed);
	layer_size = atoi(argv[1]);
	BPNN *net = bpnn_create(layer_size, 16, 1);
	float out_err;
	float hid_err;

	//printf("Problem set up");
	load(net);
	//printf("input initialized");
	bpnn_train(net, &out_err, &hid_err);
	//printf("***********result***********\n");
	//printf("Output Error = %f\n",out_err);
	//printf("Hidden Error = %f\n",hid_err);
	bpnn_save(net, "out.txt");
	bpnn_free(net);
}
