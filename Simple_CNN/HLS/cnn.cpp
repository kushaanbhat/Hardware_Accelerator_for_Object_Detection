#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "cnn.h"
#include "cnn_weights.h"
#include <cmath> // For ReLU, softmax, and other operations

// HLS CNN Function with AXI Stream Interface
void cnn(hls::stream<AXI_VAL> &image_in, hls::stream<AXI_VAL> &predict_probability, hls::stream<AXI_VAL> &predict_class) {
#pragma HLS INTERFACE axis port=image_in
#pragma HLS INTERFACE axis port=image_out
#pragma HLS INTERFACE s_axilite port=return

    // Define input image dimensions
    float image[img_h][img_w][img_d];
    float c_out[c_out_h][c_out_w][c_out_d];
    float m_out[m_out_h][m_out_w][m_out_d];
    float f_out[f_out_h];
    float d_out[d_out_h];


    // Read input image from AXI stream
    for (int i = 0; i < img_h; i++) {
        for (int j = 0; j < img_w; j++) {
            for (int k = 0; k < img_d; k++) {
#pragma HLS PIPELINE
                AXI_VAL temp_image = image_in.read();
                image[i][j][k] = axi_to_float(temp_image);
                //image[i][j][k] = image[i][j][k] / 255.0f; // Normalize to [0, 1]
            }
        }
    }

    // Conv2D Layer
    conv2d(image, c_out);
    maxpool2d(c_out, m_out);
    flatten(m_out, f_out);
    dense(f_out, d_out);
    float maxVal = d_out[0];
    int maxID = 0;
    for (int i = 1; i < d_out_h; i++) {
    	if (d_out[i] > maxVal) {
    		maxVal = d_out[i];
    		maxID = i;
    	}
     }

    predict_probability.write(float_to_axi(maxVal));
    predict_class.write(float_to_axi(maxID));

/*
    // Write the result matrix to AXI stream
    for (int k = 0; k < m_out_d; k++) {
    	for (int i = 0; i < m_out_h; i++) {
    		for (int j = 0; j < m_out_w; j++) {
#pragma HLS PIPELINE
                image_out.write(float_to_axi(m_out[i][j][k]));
            }
        }
    }
    */
}

// Conv2D Layer Definition
void conv2d(float image[img_h][img_w][img_d], float c_out[c_out_h][c_out_w][c_out_d]) {

    for (int f = 0; f < c_out_d; f++) {
        #pragma HLS UNROLL
        for (int i = 0; i < c_out_h; i++) {
            for (int j = 0; j < c_out_w; j++) {
                #pragma HLS PIPELINE
                float sum = biases[f];
                for (int c = 0; c < img_d; c++) {
                    for (int ki = 0; ki < w_1_h; ki++) {
                        for (int kj = 0; kj < w_1_w; kj++) {
                                sum += image[i+ki][j+kj][c] * weight[ki][kj][c][f];
                        }
                    }
                }
                c_out[i][j][f] = (sum > 0) ? sum : 0; // ReLU Activation
            }
        }
    }
}


// Max Pooling Layer Definition
void maxpool2d(float c_out[c_out_h][c_out_w][c_out_d], float m_out[m_out_h][m_out_w][m_out_d]) {
    for (int f = 0; f < m_out_d; f++) {
        for (int i = 0; i < m_out_h; i++) {
            for (int j = 0; j < m_out_w; j++) {
#pragma HLS PIPELINE
                float max_val = -FLT_MAX;
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
void flatten(float m_out[m_out_h][m_out_w][m_out_d], float f_out[f_out_h]) {
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
void dense(float f_out[f_out_h], float d_out[d_out_h]) {
    for (int i = 0; i < d_out_h; i++) {
#pragma HLS PIPELINE
        float sum = dense_biases[i];
        for (int j = 0; j < f_out_h; j++) {
            sum += f_out[j] * dense_weights[j][i];
        }
        d_out[i] = fmaxf(0.0f, sum); // ReLU Activation
    }
}
