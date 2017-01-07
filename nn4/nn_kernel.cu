#include <stdio.h>
#include <math.h>
#include "support.cu"

#define TILE_SIZE 64
#define BLOCK_SIZE 64


__global__ void sgemm(int m, int n, float* in, float* w, float* out) {

	//__shared__ float d_IN[TILE_SIZE*2];
	//__shared__ float d_W[TILE_SIZE][16];
	__shared__ float partialSum[16][2*TILE_SIZE];
	int b = blockIdx.x;
	int t = threadIdx.x;
	int start = b*blockDim.x*2+t;
	if (start == 0) {
		in[0] = 1.0;
		//for (int l = 1; l <= n; l++) {
		//	out[l] = 0.0;
		//}
	}
	//int c = 0;
	//for (; c < (m-1)/(2*TILE_SIZE) + 1; c++) {
	//	float s1, s2;
	if (start < m) {
		if (b != (m-1)/(2*TILE_SIZE)) {
			float cur1 = in[start];
			float cur2 = in[start+blockDim.x];
			for (int i = 0; i < n; i++) {			
				partialSum[i][t] = cur1 * w[i*n+(start)];
				partialSum[i] [t+blockDim.x]= cur2 * w[i*n+(start + blockDim.x)];
				//w[(start+t)*n+i] = -1;
			}
			__syncthreads();
			for (int stride = blockDim.x; stride > 0; stride /= 2) {
				//if (start + stride < m) {
				//float cur2 = in[start + stride];
				if (t < stride) {
					for (int k = 0; k < n; k++) {
						//partialSum[t + stride][k] = cur2*w[(start + stride)*n+k];
						//w[(start+t+stride)*n+k] = -1;
						//__syncthreads();
						partialSum[k][t] += partialSum[k][t+stride];
					}
				}
				//}

				__syncthreads();
			}
		}
		else {
			if (start + BLOCK_SIZE < m) {
				float cur1 = in[start];
				float cur2 = in[start+BLOCK_SIZE];
				for(int k = 0; k < n; k++) {
					partialSum[k][t] = cur1*w[k*n+start] + cur2*w[k*n+(start+BLOCK_SIZE)];
				}
			}
			else {
				float cur1 = in[start];
				for (int k = 0; k < n; k++) {
					partialSum[k][t] = cur1*w[k*n+start];
				}
			}
			__syncthreads();
			if (t == 0) {
				for (int j = 1; j + start < m && j < BLOCK_SIZE; j++) {
					for (int k = 0; k < n; k++) {
						partialSum[k][0] += partialSum[k][j];
					}
				}
			} 	

		}
	}
	__syncthreads();
	if (t == 0) {
		for (int p = 0; p < n; p++) {
			//printf("block%d: add %f to out %d\n",b,partialSum[0][p],p+1);
			atomicAdd(&(out[p+1]),partialSum[p][0]);
		}
	}

	/*
	   for (int j = 0; j < TILE_SIZE; j++) {
	   if (tx < n && c*TILE_SIZE+j < m) {
	   tmpsum += d_IN[j] * w[(c*TILE_SIZE+j)*n+t];
//w[(c+j)*n+t] = -1;
}

}
__syncthreads();		
	 */
//}
__syncthreads();

/*
   if (b == (m-1)/BLOCK_SIZE && t == 0) {
   printf("calculating sigmoid\n");
   for (int q = 1; q <= n; q++) {
   out[q] = (1.0/(1.0+exp(-out[q])));
   }
   }
 */
}

__global__ void sigmoid(int n, float* in, float* out) {

	int index = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	if (index < n) {
		out[index+1] = (1.0/(1.0+exp(-in[index+1])));
	}
}

__global__ void output_error(float* delta, float* target, float* output, int nj) {
	int b = blockIdx.x;
	int t = threadIdx.x;
	int index = b*BLOCK_SIZE + t;
	if (index < nj) {
		if (index == 0) {
			delta[0] = 0.0;
			//err = 0;
		}
		//for (int j = 0; j < nj; j++) {
		float result = output[index+1];
		float t = target[index+1];
		float tmp = result * (1.0 - result) * (t - result);
		delta[index+1] = tmp;
		//printf("delta %d : %f\n",index+1,delta[index+1]);
		//float errtmp = abs(delta[index+1]);
		//atomicAdd(err,errtmp);
	}
	}

	__global__ void hidden_error(float* delta_h, int nh, float* delta_o, int no, float* who, float* hidden) {
		int b = blockIdx.x;
		int t = threadIdx.x;
		int index = b*BLOCK_SIZE + t;
		if (index < nh) {
			float h = hidden[index+1];
			float sum = 0.0;
			for (int j = 1; j <= no; j++) {
				sum += delta_o[j] * who[(j-1)*no+index+1];
			}
			delta_h[index+1] = h*(1.0-h)*sum;
			//atomicAdd(err, abs(delta_h[index+1]));
		}
	}

	__global__ void adjust_weights(float* delta, int ndelta, float* ly, int nly, float* w, float* oldw) {
		//delta is the delta on the next layer
		//ly is units of the current layer
		__shared__ float tmpdelta[TILE_SIZE];
		//__shared__ float tmply[TILE_SIZE];

		int b = blockIdx.x;
		int t = threadIdx.x;
		int i1 = b*BLOCK_SIZE + t;
		if (t == 0) {
			if (i1 == 0) {
				ly[0] = 1.0;
			}
			for (int i = 0; i < ndelta; i++) {
				tmpdelta[i] = delta[i+1];
			}
		}
		__syncthreads();	
		/*
		   for (int c = 0; c < nly/TILE_SIZE+1; c++) {
		   int i1 = c*TILE_SIZE+t;

		   if (t < ndelta) {
		   tmpdelta[t] = delta[t+1];
		   }
		   else {
		   tmpdelta[t] = 0.0;
		   }
		 */
		if (i1 <= nly) {
			float tmpy = ly[i1];
			for (int j = 0; j < ndelta; j++) {
				//if (c*TILE_SIZE+j <= nly && t < ndelta) {
				int i2 =  j*ndelta +i1;
				float new_w = ((ETA * tmpdelta[j] * tmpy) + (MOMENTUM * oldw[i2]));
				w[i2] += new_w;
				oldw[i2] = new_w;
				//}
			}
		}

		__syncthreads();
		/*
		   if (index < ndelta) {
		   if(index == 0) {
		   ly[0] = 1.0;
		   }
		//float tmpdelta = delta[index+1];
		for (int k = 0; k <= nly; k++) {
		int i = index + ndelta*k;
		float new_w = ((ETA * delta[index+1] * ly[k] ) + (MOMENTUM * oldw[i]));
		w[i] += new_w;
		oldw[i] = new_w;
		}
		}
		 */
	}

	void mat2vec(float** mat, float* w, float* b, int m, int n) {
		//float* vec;
		//vec = (float*) malloc(sizeof(float)*m*n);
		for (int i = 0; i < m; i++) {
			b[i] = mat[i][0];
			for (int j = 1; j < n; j++) {
				w[i*(n-1)+j-1] = mat[i][j];
			}
		}
		//return vec;
	}

	void mat2vec_mod(float** mat, float* w, float* b, int m, int n) {
		//float* vec;
		//vec = (float*) malloc(sizeof(float)*m*n);
		for (int i = 0; i < m; i++) {
			b[i] = mat[i][0];
			for (int j = 1; j < n; j++) {
				w[(j-1)*(n-1)+i] = mat[i][j];
			}
		}
		//return vec;
	}


	void vec2mat(float** mat, float* w, float* b, int m, int n) {
		for (int i = 0; i < m; i++) {
			mat[i][0] = b[i];
			for (int j = 1; j < n; j++) {
				mat[i][j] = w[i*(n-1)+j-1];
			}
		}
	}

	void vec2mat_mod(float** mat, float* w, float* b, int m, int n) {
		for (int i = 0; i < m; i++) {
			mat[i][0] = b[i];
			for (int j = 1; j < n; j++) {
				mat[i][j] = w[(j-1)*(n-1)+i];
			}
		}
	}
	void test_save(float* vec, int size, const char* filename) {
		FILE *pFile;
		pFile = fopen(filename, "w+");
		for (int k = 0; k < size; k++) {
			fprintf(pFile, "%f\n",vec[k]);
		}
		fclose(pFile);
	}

	void test_printVec(float* vec, int size) {
		for (int t = 0; t < size; t++) {
			printf("%f\n",vec[t]);
		}
	}

	float getError(float* delta, int dsize) {
		float sum;
		sum = 0.0;
		for (int i = 0; i < dsize; i++) {
			sum += abs(delta[i]);
		}
		return sum;	
	}


	void bpnn_train(BPNN* net, float* eo, float* eh) {
		int in, hid, out;
		in = net->input_n;
		hid = net->hidden_n;
		out = net->output_n;
		int vec_inwsz = (in+1)*hid;
		int vec_hwsz = (hid + 1) * out;
		Timer timer;
		//*eo = 0.0;
		//*eh = 0.0;
		//printf("conveting w to vector ...");
		//startTime(&timer);
		float* inb = (float*) malloc(sizeof(float)*(in+1));
		float* inoldb = (float*) malloc(sizeof(float)*(in+1));
		float* holdb = (float*) malloc(sizeof(float)*(hid+1));
		float* vec_inoldw;
		//using pin memory for weight
		cudaHostAlloc((void **) &vec_inoldw, sizeof(float)*vec_inwsz, cudaHostAllocDefault);
		float* vec_holdw;
		cudaHostAlloc((void **) &vec_holdw, sizeof(float)*vec_hwsz, cudaHostAllocDefault); 
		float* hb = (float*) malloc(sizeof(float)*(hid+1));
		float* vec_inw;
		cudaHostAlloc((void **) &vec_inw, sizeof(float)*vec_inwsz, cudaHostAllocDefault);
		mat2vec_mod(net->input_weights, vec_inw, inb, in+1, hid+1);
		float* vec_hw;
		cudaHostAlloc((void **) &vec_hw, sizeof(float)*vec_hwsz,cudaHostAllocDefault);
		mat2vec_mod(net->hidden_weights, vec_hw, hb, hid+1, out+1);
		mat2vec_mod(net->input_prev_weights, vec_inoldw, inoldb, in+1, hid+1);
		mat2vec_mod(net->hidden_prev_weights, vec_holdw, holdb, hid+1, out+1);
		//stopTime(&timer);
		//printf("%f s\n",elapsedTime(timer));
		//device variables for input to hidden
		float* inu_d, *inw_d, *hidu_d, *hidutmp;
		//device variables for hidden to output
		float* hw_d, *outu_d, *oututmp;
		//device variable for errors
		float* hdelta_d, *odelta_d, *target_d;
		//device variables for ws
		float* inoldw_d, *hidoldw_d;
		//define streams
		
		printf("allocating device memory ...");
		startTime(&timer);

		cudaMalloc((void**) &inu_d, sizeof(float)*(in+1));
		cudaMalloc((void**) &inw_d, sizeof(float)*vec_inwsz);
		cudaMalloc((void**) &hidu_d, sizeof(float)*(hid+1));
		cudaMalloc((void**) &target_d, sizeof(float)*(out+1));
		cudaMalloc((void**) &hdelta_d, sizeof(float)*(hid+1));
		cudaMalloc((void**) &odelta_d, sizeof(float)*(out+1));
		cudaMalloc((void**) &inoldw_d, sizeof(float)*vec_inwsz);
		cudaMalloc((void**) &hidoldw_d, sizeof(float)*vec_hwsz);
		cudaMalloc((void**) &hw_d, sizeof(float)*vec_hwsz);
		cudaMalloc((void**) &outu_d, sizeof(float)*(out+1));

		cudaMalloc((void**) &hidutmp, sizeof(float)*(hid+1));
		cudaMalloc((void**) &oututmp, sizeof(float)*(out+1));
		stopTime(&timer);
		printf("%f s\n",elapsedTime(timer));
		printf("copying data from host to device ...");
		startTime(&timer);
		cudaMemcpy(inu_d, net->input_units, sizeof(float)*(in+1), cudaMemcpyHostToDevice);
		cudaMemcpy(inw_d, vec_inw, sizeof(float)*vec_inwsz, cudaMemcpyHostToDevice);
		cudaMemcpy(hw_d, vec_hw, sizeof(float)*vec_hwsz, cudaMemcpyHostToDevice);
		cudaMemcpy(target_d, net->target, sizeof(float)*(out+1), cudaMemcpyHostToDevice);
		cudaMemcpy(inoldw_d, vec_inoldw, sizeof(float)*vec_inwsz, cudaMemcpyHostToDevice);
		cudaMemcpy(hidoldw_d, vec_holdw, sizeof(float)*vec_hwsz, cudaMemcpyHostToDevice);
		stopTime(&timer);
		printf("%f s\n",elapsedTime(timer));
		//launching kernels

		//printf("forward propagation ...\n");
		//fflush(stdout);
		startTime(&timer);
		dim3 DimBlock(BLOCK_SIZE,1,1);
		dim3 DimGrid((hid - 1)/BLOCK_SIZE + 1,1,1);
		//printf("------calculating hidden units ...\n");
		dim3 DimBlock1(BLOCK_SIZE,1,1);
		dim3 DimGrid1(in/BLOCK_SIZE + 1,1,1);
		dim3 DimGridRed1(in/(BLOCK_SIZE*2)+1,1,1);
		sgemm<<<DimGridRed1, DimBlock1>>>(in+1, hid, inu_d, inw_d, hidutmp);
		sigmoid<<<DimGrid, DimBlock>>>(hid, hidutmp,hidu_d);

		//test code
		//float* hiduz_d_test = (float *)malloc((hid+1)*sizeof(float));
		//cudaMemcpy(hiduz_d_test, hidu_d, (hid+1)*sizeof(float), cudaMemcpyDeviceToHost);
		//test_printVec(hiduz_d_test, hid+1);
		/*
		   printf("[debug]inspect w:\n");
		   float* tmpw_test = (float*)malloc(vec_inwsz*sizeof(float));
		   cudaMemcpy(tmpw_test, inw_d, vec_inwsz*sizeof(float), cudaMemcpyDeviceToHost);
		   test_printVec(tmpw_test, vec_inwsz);
		   printf("[debug] finish w\n");
		 */
		//printf("debug: hiduz_size = %d, hid = %d\n",sizeof(hidu_d)/sizeof(float),hid);	
		dim3 DimBlock2(BLOCK_SIZE,1,1);
		dim3 DimGrid2((hid)/BLOCK_SIZE + 1,1,1);
		dim3 DimGridRed2(hid/(BLOCK_SIZE*2)+1,1,1);
		dim3 DimBlock3(BLOCK_SIZE,1,1);
		dim3 DimGrid3((out-1)/BLOCK_SIZE+1,1,1);
		//printf("------calculating output units ...\n");
		sgemm<<<DimGridRed2, DimBlock2>>>(hid + 1, out, hidu_d, hw_d, oututmp);
		sigmoid<<<DimGrid3, DimBlock3>>>(out,oututmp,outu_d);
		//test code
		//float* outuz_d_test = (float*)malloc((out+1)*sizeof(float));
		//cudaMemcpy(outuz_d_test, outu_d, (out+1)*sizeof(float), cudaMemcpyDeviceToHost);
		//test_printVec(outuz_d_test,out+1);
		//printf("target:\n");
		//test_printVec(net->target, out+1);

		//printf("calculating error ...\n");
		//printf("------output error ...\n");
		output_error<<<DimGrid2, DimBlock2>>>(odelta_d, target_d, outu_d, out);
		//TEST CODE
		//float* odelta_d_test = (float*)malloc((out+1)*sizeof(float));
		//cudaMemcpy(odelta_d_test, odelta_d, (out+1)*sizeof(float),cudaMemcpyDeviceToHost);
		//test_printVec(odelta_d_test, out+1);
		//printf("------hidden error ...\n");
		hidden_error<<<DimGrid, DimBlock>>>(hdelta_d, hid, odelta_d, out, hw_d, hidu_d);
		//float* hdelta_d_test = (float*)malloc((hid+1)*sizeof(float));
		//cudaMemcpy(hdelta_d_test, hdelta_d, (hid+1)*sizeof(float),cudaMemcpyDeviceToHost);
		//test_printVec(hdelta_d_test, hid+1);
		//printf("updating weights ...\n");
		//printf("------hidden2out ...\n");
		adjust_weights<<<DimGrid2, DimBlock2>>>(odelta_d, out, hidu_d, hid, hw_d, hidoldw_d);
		//TEST CODE
		//float* hw_d_test = (float*)malloc(vec_hwsz*sizeof(float));
		//cudaMemcpy(hw_d_test, hw_d, vec_hwsz*sizeof(float),cudaMemcpyDeviceToHost);

		//test_printVec(hw_d_test, vec_hwsz);

		//printf("------in2hidden ...\n");
		adjust_weights<<<DimGrid1, DimBlock1>>>(hdelta_d, hid, inu_d, in, inw_d, inoldw_d);
		//test code
		//float* inw_d_test = (float*)malloc(vec_inwsz*sizeof(float));
		//cudaMemcpy(inw_d_test, inw_d, vec_inwsz*sizeof(float),cudaMemcpyDeviceToHost);
		//test_printVec(inw_d_test, vec_inwsz);

		stopTime(&timer);
		printf("Total kernel execution time: %f s\n",elapsedTime(timer));
		printf("copying to host ...");
		startTime(&timer);
		cudaMemcpy(net->hidden_units, hidu_d, sizeof(float)*(hid+1), cudaMemcpyDeviceToHost);
		cudaMemcpy(net->output_units, outu_d, sizeof(float)*(out+1), cudaMemcpyDeviceToHost);
		cudaMemcpy(net->hidden_delta, hdelta_d, sizeof(float)*(hid+1), cudaMemcpyDeviceToHost);
		cudaMemcpy(net->output_delta, odelta_d, sizeof(float)*(out+1), cudaMemcpyDeviceToHost);

		cudaMemcpy(vec_inw, inw_d, sizeof(float)*vec_inwsz, cudaMemcpyDeviceToHost);
		cudaMemcpy(vec_hw, hw_d, sizeof(float)*vec_hwsz, cudaMemcpyDeviceToHost);
		cudaMemcpy(vec_inoldw, inoldw_d, sizeof(float)*vec_inwsz, cudaMemcpyDeviceToHost);
		cudaMemcpy(vec_holdw, hidoldw_d, sizeof(float)*vec_hwsz, cudaMemcpyDeviceToHost);
		stopTime(&timer);
		printf("%f s\n",elapsedTime(timer));
		//test_save(vec_hw, vec_hwsz, "hidden_weight_vector.txt");
		vec2mat_mod(net->input_weights, vec_inw, inb, in+1, hid+1);
		vec2mat_mod(net->hidden_weights, vec_hw, hb, hid+1, out+1);
		vec2mat_mod(net->input_prev_weights, vec_inoldw, inoldb, in+1, hid+1);
		vec2mat_mod(net->hidden_prev_weights, vec_holdw, holdb, hid+1, out+1);
		*eo = getError(net->output_delta, out + 1);
		*eh = getError(net->hidden_delta, hid + 1);	
		printf("freeing variables ...\n");
		cudaFree(inu_d);
		cudaFree(inw_d);
		cudaFree(hidu_d);
		cudaFree(hw_d);
		cudaFree(outu_d);
		cudaFree(hdelta_d);
		cudaFree(odelta_d);
		cudaFree(target_d);
		cudaFree(inoldw_d);
		cudaFree(hidoldw_d);
		cudaFree(hidutmp);
		cudaFree(oututmp);

		cudaFreeHost(vec_inw);
		cudaFreeHost(vec_hw);	
		cudaFreeHost(vec_inoldw);
		cudaFreeHost(vec_holdw);

		free(inb);
		free(hb);
		free(inoldb);
		free(holdb);

		//FREE TEST VARIABLES
		//free(hiduz_d_test);
		//free(outuz_d_test);
		//free(odelta_d_test);
		//free(hw_d_test);
		//free(inw_d_test);
	}



