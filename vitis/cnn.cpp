#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <cmath>
#include "dataTypes.h"
#include "cnn_weights_16bits.h"

void cnn(hls::stream<AXI_VAL> &image_in, hls::stream<AXI_VAL> &probability_out, hls::stream<AXI_VAL> &class_out) {
#pragma HLS INTERFACE mode=axis register_mode=both depth=2700 port=image_in register=true
#pragma HLS INTERFACE mode=axis register_mode=both depth=43 port=probability_out register=true
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=class_out register=true
#pragma HLS INTERFACE mode=s_axilite port=return

    data image[img_h][img_w][img_d], c_out[c_out_h][c_out_w][c_out_d], m_out[m_out_h][m_out_w][m_out_d], f_out[f_out_h], d_out[d_out_h], s_out[s_out_h];
    int MaxID;

	for (int i = 0; i < img_h; i++) {
		for (int j = 0; j < img_w; j++) {
			for (int k = 0; k < img_d; k++) {
#pragma HLS PIPELINE
				AXI_VAL temp = image_in.read();
				image[i][j][k] = axi_to_fixed(temp);
			}
		}
	}

	conv2d(image, c_out);
	maxpool2d(c_out, m_out);
	flatten(m_out, f_out);
	dense(f_out, d_out);
	softmax(d_out, s_out);
	MaxID = 0;
	data max_value = s_out[0];

	for (int i = 1; i < s_out_h; i++) {
		if (s_out[i] > max_value) {
			max_value = s_out[i];
			MaxID = i;
		}
	}

	for (int i = 0; i < d_out_h; i++) {
		//for (int j = 0; j < m_out_w; j++) {
			//for (int k = 0; k < m_out_d; k++) {
#pragma HLS PIPELINE
		AXI_VAL temp = fixed_to_axi(s_out[i]);
		if (i == d_out_h-1) temp.last = 1; // Set last bit for AXI compliance
		probability_out.write(temp);
			//}
		//}
	}
	 AXI_VAL class_val = float_to_axi(MaxID);
	 class_val.last = 1;
	 class_out.write(class_val);
}

// Conv2D Layer Definition
void conv2d(data image[img_h][img_w][img_d], data c_out[c_out_h][c_out_w][c_out_d]) {

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
void dense(data f_out[f_out_h], data d_out[d_out_h]) {
    for (int i = 0; i < d_out_h; i++) {
#pragma HLS UNROLL
        data sum = d_biases[i];
        for (int j = 0; j < f_out_h; j++) {
            sum += f_out[j] * d_weights[j][i];
        }
        d_out[i] = sum; // ReLU Activation
    }
}

void softmax(data d_out[d_out_h], data s_out[s_out_h]) {
    data max = d_out[0];
    for (int i = 1; i < s_out_h; i++) {
        if (d_out[i] > max) max = d_out[i];
    }

    data sum = 0.0;
    for (int i = 0; i < s_out_h; i++) {
        s_out[i] = expf(d_out[i] - max); // for numerical stability
        sum += s_out[i];
    }

    for (int i = 0; i < s_out_h; i++) {
        s_out[i] /= sum;
    }
}


