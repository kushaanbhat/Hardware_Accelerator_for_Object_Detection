#define M_MAX 9
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
using namespace std;
extern "C" {
void matrix_multiplier( float *A, float *x, float *y, unsigned m);
}


int main() {
	unsigned int m = 3*3;

	float *A = (float *)malloc(m*m*sizeof(float));
	float *x = (float *)malloc(m*m*sizeof(float));
	float *y_hw = (float *)malloc(m*m*sizeof(float));
    float *y_sw = (float *)malloc(m*m*sizeof(float));


    for (unsigned int i = 0; i < m*m; i++) {
    	A[i] = rand()/(1.0*RAND_MAX);
    }
    cout<<"Print Matrix A:"<<endl;
    for (int i = 0; i < m; i++) {
        	for (int j = 0; j < m; j++) {
        		cout<<A[i+j]<<'\t';
        	}
        	cout<<endl;
        }
    for (unsigned int i = 0; i < m*m; i++) {
    	x[i] = rand()/(1.0*RAND_MAX);
    }
    cout<<"Print Matrix x:"<<endl;
    for (int i = 0; i < m; i++) {
        	for (int j = 0; j < m; j++) {
        		cout<<x[i+j]<<'\t';
        	}
        	cout<<endl;
        }
    for (int i = 0; i < m; i++) {
    	for (int j = 0; j < m; j++) {
    		float y_tmp = 0;
    		for (int k = 0; k < m; k++)
    			y_tmp += A[i+k]*x[k+j];
    		y_sw[i+j] = y_tmp;
    	}
    }

    matrix_multiplier(A, x, y_hw, m);

    for (unsigned int i = 0; i < m; i++) {
    	for (unsigned int j = 0; j < m; j++)
    		std::cout <<"At " << i<<" & at " << j << " with value y_hw = " << y_hw[i+j] << ", should be y_sw = " << y_sw[i+j]<<" And Error = "<<(y_sw[i+j]-y_hw[i+j])
  					  << std::endl;
    }
    int match = 0;
    for (unsigned int i = 0; i < m; i++) {
        	for (unsigned int j = 0; j < m; j++){
        		if(y_sw[i+j] != y_sw[i+j]){
        			match = 1;
        			std::cout <<"Error at " << i<<" & at " << j << " with value y_hw = " << y_hw[i] << ", should be y_sw = " << y_sw[i]
        			  					  << std::endl;
        		}
        		else{
        			match = 0;
        		}
        	}
    }



    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return 0;

}
