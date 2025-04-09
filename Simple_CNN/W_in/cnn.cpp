#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "dataTypes.h"
#include <cmath> // For ReLU, softmax, and other operations

// HLS CNN Function with AXI Stream Interface
void cnn(hls::stream<AXI_VAL> &image_in, hls::stream<AXI_VAL> &c_weights_in, hls::stream<AXI_VAL> &c_biases_in, hls::stream<AXI_VAL> &d_weights_in, hls::stream<AXI_VAL> &d_biases_in, hls::stream<AXI_VAL> &predict_class) {
#pragma HLS INTERFACE mode=axis register_mode=both port=image_in
#pragma HLS INTERFACE mode=axis register_mode=both port=c_weights_in
#pragma HLS INTERFACE mode=axis register_mode=both port=c_biases_in
#pragma HLS INTERFACE mode=axis register_mode=both port=d_weights_in
#pragma HLS INTERFACE mode=axis register_mode=both port=d_biases_in
#pragma HLS INTERFACE mode=axis register_mode=both port=predict_class
#pragma HLS INTERFACE s_axilite port=return

    // Define input image dimensions
    data image[img_h][img_w][img_d];
    data weight[3][3][3][32];
    data biases[32];
    data c_out[c_out_h][c_out_w][c_out_d];
    data m_out[m_out_h][m_out_w][m_out_d];
    data f_out[f_out_h];
    data dense_weights[6272][43];
    data dense_biases[43];
    data d_out[d_out_h];


    // Read input image from AXI stream
    for (int i = 0; i < img_h; i++) {
        for (int j = 0; j < img_w; j++) {
            for (int k = 0; k < img_d; k++) {
#pragma HLS PIPELINE
                AXI_VAL temp_image = image_in.read();
                image[i][j][k] = axi_to_fixed(temp_image);
            }
        }
    }

    for (int f = 0; f < c_out_d; f++) {
    	for (int c = 0; c < img_d; c++) {
    		for (int ki = 0; ki < w_1_h; ki++) {
    			for (int kj = 0; kj < w_1_w; kj++) {
#pragma HLS PIPELINE
                AXI_VAL temp_weights = c_weights_in.read();
                weight[ki][kj][c][f]= axi_to_fixed(temp_weights);
    			}
    		}
    	}
    }

    for (int i = 0; i < c_out_d; i++) {
    #pragma HLS PIPELINE
    	AXI_VAL temp_biases = c_biases_in.read();
    	biases[i] = axi_to_fixed(temp_biases);
    }

    // Read input of Dense Weights
    for (int i = 0; i < f_out_h; i++) {
    	for (int j = 0; j < d_out_h; j++) {
#pragma HLS PIPELINE
    		AXI_VAL temp_dense_weights = d_weights_in.read();
    		dense_weights[i][j] = axi_to_fixed(temp_dense_weights);
    	}
    }

    for (int i = 0; i < d_out_h; i++) {
#pragma HLS PIPELINE
        	AXI_VAL temp_dense_biases = d_biases_in.read();
        	dense_biases[i] = axi_to_fixed(temp_dense_biases);
    }

    // Conv2D Layer
    conv2d(image, c_out, weight, biases);
    // MaxPooling Layer
    maxpool2d(c_out, m_out);
    // Flatten Layer
    flatten(m_out, f_out);
    // Dense Layer
    dense(f_out, d_out, dense_weights, dense_biases);
    data maxVal = d_out[0];
    int maxID = 0;
    for (int i = 1; i < d_out_h; i++) {
    	if (d_out[i] > maxVal) {
    		maxVal = d_out[i];
    		maxID = i;
    	}
     }

    predict_class.write(fixed_to_axi(maxID));

}

// Conv2D Layer Definition
void conv2d(data image[30][30][3], data c_out[28][28][32], const data weight[3][3][3][32], const data biases[32]) {

    for (int f = 0; f < c_out_d; f++) {
        for (int i = 0; i < c_out_h; i++) {
            for (int j = 0; j < c_out_w; j++) {
                #pragma HLS PIPELINE
                data sum = biases[f];
                data zero = 0;
                for (int c = 0; c < img_d; c++) {
                    for (int ki = 0; ki < w_1_h; ki++) {
                        for (int kj = 0; kj < w_1_w; kj++) {
                                sum += image[i+ki][j+kj][c] * weight[ki][kj][c][f];
                        }
                    }
                }
                c_out[i][j][f] = (sum > zero) ? sum : zero; // ReLU Activation
            }
        }
    }
}


// Max Pooling Layer Definition
void maxpool2d(data c_out[c_out_h][c_out_w][c_out_d], data m_out[m_out_h][m_out_w][m_out_d]) {
    for (int f = 0; f < m_out_d; f++) {
        for (int i = 0; i < m_out_h; i++) {
            for (int j = 0; j < m_out_w; j++) {
#pragma HLS PIPELINE
                data max_val = -FLT_MAX;
                for (int ki = 0; ki < 2; ki++) {
                    for (int kj = 0; kj < 2; kj++) {
                        max_val = fmaxf(max_val, c_out[2 * i + ki][2 * j + kj][f]);
                    }
                }
                m_out[i][j][f] = max_val;
            }
        }
    }
}

// Flatten Layer Definition
void flatten(data m_out[m_out_h][m_out_w][m_out_d], data f_out[f_out_h]) {
    for (int i = 0; i < m_out_h; i++) {
        for (int j = 0; j < m_out_w; j++) {
            for (int f = 0; f < m_out_d; f++) {
#pragma HLS PIPELINE
            	f_out[i * m_out_h * m_out_d + j * m_out_d + f] = m_out[i][j][f];
            }
        }
    }
}


// Dense Layer Definition
void dense(data f_out[f_out_h], data d_out[d_out_h], const data dense_weights[f_out_h][d_out_h], const data dense_biases[d_out_h]) {
    for (int i = 0; i < d_out_h; i++) {
#pragma HLS UNROLL
        data sum = dense_biases[i];
        for (int j = 0; j < f_out_h; j++) {
            sum += f_out[j] * dense_weights[j][i];
        }
        d_out[i] = fmaxf(0.0f, sum); // ReLU Activation
    }
}
