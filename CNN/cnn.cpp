#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "cnn.h"
<<<<<<< HEAD
#include "cnn_weights.h"
#include <cmath> // For ReLU, softmax, and other operations
=======
#include "conv1.h"
#include "conv2.h"
#include "conv3.h"
#include "conv4.h"
#include "dense1.h"
#include "dense2.h"
#include <cmath> // For ReLU, softmax, and other operations
#include <cfloat>
#include <iostream> // For debugging output
#include <fstream>  // For saving outputs to files
>>>>>>> 230aa7bf74dbbd213528fd1a18ff58b0c6c4fdf9

// HLS CNN Function with AXI Stream Interface
void cnn(hls::stream<AXI_VAL> &image_in,  hls::stream<AXI_VAL> &predict_class) {
#pragma HLS INTERFACE mode=axis register_mode=both depth=2700 port=image_in register
#pragma HLS INTERFACE mode=axis register_mode=both depth=1 port=predict_class register
#pragma HLS INTERFACE s_axilite port=return

<<<<<<< HEAD
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
=======
// Function to read input data from AXI stream and normalize
void read_input(axi_stream &input_stream, float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS]) {
    for (int i = 0; i < INPUT_HEIGHT; i++) {
        for (int j = 0; j < INPUT_WIDTH; j++) {
            for (int c = 0; c < INPUT_CHANNELS; c++) {
                #pragma HLS PIPELINE
            	fixed_t temp = input_stream.read();
                input[i][j][c] = temp; // Normalize pixel values
>>>>>>> 230aa7bf74dbbd213528fd1a18ff58b0c6c4fdf9
            }
        }
    }

<<<<<<< HEAD
    // Conv2D Layer
    conv2d(image, c_out);
    // MaxPooling Layer
    maxpool2d(c_out, m_out);
    // Flatten Layer
    flatten(m_out, f_out);
    // Dense Layer
    dense(f_out, d_out);
    float maxVal = d_out[0];
=======
/*void save_layer_output(const char* filename, float* output, int size) {
    ofstream file(filename);
    for (int i = 0; i < size; i++) {
        file << output[i] << "\n";
    }
    file.close();
}*/

// Function to print and save layer outputs
/*void print_and_save_output(const char* layer_name, float* output, int size) {
    cout << "\n" << layer_name << " output (first 10 values):\n";
    for (int i = 0; i < min(10, size); i++) {
        cout << output[i] << " ";
    }
    cout << "\n";
    save_layer_output(layer_name, output, size);
}*/

/*// Function to print layer outputs for debugging
void print_layer_output(const char* layer_name, float* output, int size) {
    cout << "\n" << layer_name << " output:\n";
    for (int i = 0; i < size; i++) {
        cout << output[i] << " ";
    }
    cout << "\n" << endl;
}*/


// Softmax function to normalize final layer outputs
void softmax(float input[FC2_OUTPUT], float output[FC2_OUTPUT]) {
    float sum_exp = 0;
    for (int i = 0; i < FC2_OUTPUT; i++) {
        output[i] = exp(input[i]);
        sum_exp += output[i];
    }
    for (int i = 0; i < FC2_OUTPUT; i++) {
        output[i] /= sum_exp;
    }
}

// Function to write output data to AXI stream with softmax
void write_output(axi_stream &output_stream, float output[FC2_OUTPUT]) {
    float softmax_output[FC2_OUTPUT];
    softmax(output, softmax_output);

    for (int i = 0; i < FC2_OUTPUT; i++) {
        #pragma HLS PIPELINE
        fixed_t temp;
        temp = softmax_output[i];
        output_stream.write(temp);
    }

    float maxVal = softmax_output[0];
>>>>>>> 230aa7bf74dbbd213528fd1a18ff58b0c6c4fdf9
    int maxID = 0;
<<<<<<< HEAD
    for (int i = 1; i < d_out_h; i++) {
    	if (d_out[i] > maxVal) {
    		maxVal = d_out[i];
    		maxID = i;
    	}
     }

    predict_class.write(float_to_axi(maxID));
=======
    for (int i = 1; i < FC2_OUTPUT; i++) {
        if (softmax_output[i] > maxVal) {
            maxVal = softmax_output[i];
            maxID = i;
        }
    }
    cout << "\nMaximum probability: " << maxVal << " for Class ID: " << maxID << "\n" << endl;
}
// Convolution Layer 1 (Conv2D with 32 filters of size 5x5)
void conv2d_1(float input[c1_w_in][c1_l_in][c1_d_in], float conv1_output[c1_w_out][c1_l_out][c1_d_out]) {
    for (int f = 0; f < c1_d_out; f++) {
        for (int i = 0; i < c1_l_out; i++) {
            for (int j = 0; j < c1_w_out; j++) {
                #pragma HLS PIPELINE
                float sum = conv1_biases[f];
                for (int c = 0; c < 3; c++) {
                    for (int ki = 0; ki < 5; ki++) {
                        for (int kj = 0; kj < 5; kj++) {
                            sum += input[i + ki][j + kj][c] * conv1_weights[f][c][ki][kj];
                        }
                    }
                }
                conv1_output[i][j][f] = fmaxf(0.0f, sum); // ReLU Activation
            }
        }
    }
}
>>>>>>> 230aa7bf74dbbd213528fd1a18ff58b0c6c4fdf9

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
#pragma HLS UNROLL
        float sum = dense_biases[i];
        for (int j = 0; j < f_out_h; j++) {
            sum += f_out[j] * dense_weights[j][i];
        }
        d_out[i] = fmaxf(0.0f, sum); // ReLU Activation
    }
}

<<<<<<< HEAD
=======
// The CNN forward function, calling all layers in sequence
void cnn(axi_stream &input_stream, axi_stream &output_stream) {
#pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    float input[30][30][3];
    read_input(input_stream, input);

    float conv1_output[26][26][32];
    conv2d_1(input, conv1_output);
    //print_and_save_output("hls_conv2d_1.csv", &conv1_output[0][0][0], c1_w_out * c1_l_out * c1_d_out);

    float conv2_output[22][22][32];
    conv2d_2(conv1_output, conv2_output);
    //print_and_save_output("hls_conv2d_2.csv", &conv2_output[0][0][0], c2_w_out * c2_l_out * c2_d_out);

    float maxpool1_output[11][11][32];
    maxpool2d_1(conv2_output, maxpool1_output);
    //print_and_save_output("hls_maxpool2d_1.csv", &maxpool1_output[0][0][0], mp1_w_out * mp1_l_out * mp1_d_out);

    float conv3_output[9][9][64];
    conv2d_3(maxpool1_output, conv3_output);
    //print_and_save_output("hls_conv2d_3.csv", &conv3_output[0][0][0], c3_w_out * c3_l_out * c3_d_out);

    float conv4_output[7][7][64];
    conv2d_4(conv3_output, conv4_output);
    //print_and_save_output("hls_conv2d_4.csv", &conv4_output[0][0][0], c4_w_out * c4_l_out * c4_d_out);

    float maxpool2_output[3][3][64];
    maxpool2d_2(conv4_output, maxpool2_output);
    //print_and_save_output("hls_maxpool2d_2.csv", &maxpool2_output[0][0][0], mp2_w_out * mp2_l_out * mp2_d_out);

    float flat_output[576];
    flatten(maxpool2_output, flat_output);
    //print_and_save_output("hls_flatten.csv", flat_output, f_out);

    float dense1_output[256];
    dense_1(flat_output, dense1_output);
    //print_and_save_output("hls_dense_1.csv", dense1_output, d1_out);

    float dense2_output[43];
    dense_2(dense1_output, dense2_output);
    //print_and_save_output("hls_dense_2.csv", dense2_output, d2_out);

    write_output(output_stream, dense2_output);
}

>>>>>>> 230aa7bf74dbbd213528fd1a18ff58b0c6c4fdf9