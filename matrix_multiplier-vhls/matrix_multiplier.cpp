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
				L5: for (int k = 0; k < m; k++)
				y_tmp += A[i+k]*x_local[k+j];
				y[i+j] = y_tmp;
			}
		}
	}
}
