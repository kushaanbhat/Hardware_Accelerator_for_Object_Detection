# Hardware_Accelerator_for_Object_Detection

This GitHub repository records the progress made during the Final Year Project for B.Tech in Electronics & Telecommunications by Asawa Madhav, Bhasme Vidhan & Bhat Kushaan under the guidance of Prof. Sagar Mhatre.

## Matrix Multiplier

The first step in developing of Object Detection Model on Hardware is the Development of a Matrix Multiplier using High-Level Synthesis.
The Code for the same is given below:

### matrix_multiplier.cpp
```c++
  #define M_MAX 9
  extern "C" {
  void matrix_multiplier( float *A, float *x, float *y, unsigned m) {
  
  	  float x_local[M_MAX];
  
  		L1: for (int i = 0; i < m; i++) {
  			L2: for (int j = 0; j < m; i++) {
  				x_local[i+j] = x[i+j];
  			}
  		}
  
  		L3: for (int i = 0; i < m; i++) {
  			L4: for (int j = 0; j < m; j++) {
  				float y_tmp = 0;
  				L5: for (int k = 0; k < m; k++){
  					y_tmp += A[i+k]*x_local[k+j];
  				}
  				y[i+j] = y_tmp;
  			}
  		}
  	}
  }
```
### matrix_multiplier_tb.cpp
```c++
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
      		for (int k = 0; k < m; k++){
      			y_tmp += A[i+k]*x[k+j];
      		}
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
          		if(((y_sw[i+j] - y_hw[i+j])>=0.5) || ((y_sw[i+j] - y_hw[i+j])<=-0.5)){
          			match = 1;
          			std::cout <<"Error at " << i<<" & at " << j << " with value y_hw = " << y_hw[i] << ", should be y_sw = " << y_sw[i]<<"Value of Match = "<<match
          			  					  << std::endl;
          		}
          		else{
          			match = 0;
          		}
          	}
      }
      std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
      return match;
}
```
