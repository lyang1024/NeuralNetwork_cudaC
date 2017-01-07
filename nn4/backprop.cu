/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *	Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */


#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include <math.h>
//#include "nn_kernel.cu"

extern int layer_size;
#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))

#define fastcopy(to,from,len)\
{\
  register char *_to,*_from;\
  register int _i,_l;\
  _to = (char *)(to);\
  _from = (char *)(from);\
  _l = (len);\
  for (_i = 0; _i < _l; _i++) *_to++ = *_from++;\
}

/*** Return random number between 0.0 and 1.0 ***/
float drnd()
{
  return ((float) rand() / (float) BIGRND);
}

/*** Return random number between -1.0 and 1.0 ***/
float dpn1()
{
  return ((drnd() * 2.0) - 1.0);
}

void load(BPNN *net) {
	float *units;
	int nr, nc, imgsize, i, j, k;
	
	nr = layer_size;

	imgsize = nr * nc;
	units = net->input_units;

	k = 1;
	for (i = 0; i < nr; i++) {
		units[k] = (float) rand()/RAND_MAX;
		k++;
	}
}
/*** The squashing function.  Currently, it's a sigmoid. ***/

/* float squash(x)
float x;
{
  float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x)));
} */


/*** Allocate 1d array of floats ***/
//need to be changed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

float *alloc_1d_dbl(int n)
{
  float *result;

  result = (float *) malloc ((unsigned) (n * sizeof (float)));
  if (result == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    return (NULL);
  }
  return (result);
}


/*** Allocate 2d array of floats ***/
//need to be changed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
float **alloc_2d_dbl(int m, int n)
{
  int i;
  float **result;

  result = (float **) malloc ((unsigned) (m * sizeof (float *)));
  if (result == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    result[i] = alloc_1d_dbl(n);
  }

  return (result);
}


void bpnn_randomize_weights(float** w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
     w[i][j] = (float) rand()/RAND_MAX;
    //  w[i][j] = dpn1();
    }
  }
}

void bpnn_randomize_row(float *w, int m)
{
	int i;
	for (i = 0; i <= m; i++) {
     //w[i] = (float) rand()/RAND_MAX;
	 w[i] = 0.1;
    }
}


void bpnn_zero_weights(float **w, int m, int n)
{
  int i, j;

  for (i = 0; i <= m; i++) {
    for (j = 0; j <= n; j++) {
      w[i][j] = 0.0;
    }
  }
}

/*
void bpnn_initialize(seed)
{
  printf("Random number generator seed: %d\n", seed);
  srand(seed);
}
*/

BPNN *bpnn_internal_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));
  if (newnet == NULL) {
    printf("BPNN_CREATE: Couldn't allocate neural network\n");
    return (NULL);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1);
  //newnet->hidden_a = alloc_1d_dbl(n_hidden + 1);
  newnet->output_units = alloc_1d_dbl(n_out + 1);
  //newnet->output_a = alloc_1d_dbl(n_hidden + 1);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1);
  newnet->output_delta = alloc_1d_dbl(n_out + 1);
  newnet->target = alloc_1d_dbl(n_out + 1);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1);

  return (newnet);
}


void bpnn_free(BPNN *net)
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  //free((char *) net->hidden_a);
  free((char *) net->output_units);
  //free((char *) net->output_a);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i <= n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i <= n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}


/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/

BPNN* bpnn_create(int n_in, int n_hidden, int n_out)
{

  BPNN *newnet;

  newnet = bpnn_internal_create(n_in, n_hidden, n_out);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return (newnet);
}
/*
void bpnn_train(BPNN* net) {
	//int in, hid, out;
	//in = net->input_n;
	//hid = net->hidden_n;
	//out = net->output_n;
	bpnn_forward(net);
}
*/

void bpnn_save(BPNN *net, const char* filename) {
	int n1 = net->input_n;
	int n2 = net->hidden_n;
	int n3 = net->output_n;
	float** inw = net->input_weights;
	float** hidw = net->hidden_weights;
	FILE *pFile;
	pFile = fopen(filename, "w+");
	fprintf(pFile, "Input Weights: %dx%d\n",n1+1,n2+1);
	for (int i = 0; i <= n1; i++) {
		for(int j = 0; j <= n2; j++) {
			fprintf(pFile, "%d,%d,%f\n",i,j,inw[i][j]);
		}
	}
	fprintf(pFile, "Hidden Weights: %dx%d\n",n2+1,n3+1);
	for (int i = 0; i <= n2; i++) {
		for (int j = 0; j <= n3; j++) {
			fprintf(pFile, "%d,%d,%f\n",i,j,hidw[i][j]);
		}
	}
	fclose(pFile);
}
