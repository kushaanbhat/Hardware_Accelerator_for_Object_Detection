#include "cnn.h"
#include "conv1.h"
#include "conv2.h"
#include "conv3.h"
#include "conv4.h"
#include "dense1.h"
#include "dense2.h"
#include <cmath> // For ReLU and other operations
#include <cfloat>

using namespace std;

// Function to read input data from AXI stream
void read_input(axi_stream &input_stream, float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS]) {
    for (int i = 0; i < INPUT_HEIGHT; i++) {
        for (int j = 0; j < INPUT_WIDTH; j++) {
            for (int c = 0; c < INPUT_CHANNELS; c++) {
                #pragma HLS PIPELINE
                axi_t temp = input_stream.read();
                input[i][j][c] = temp.data;
            }
        }
    }
}

// Function to write output data to AXI stream
void write_output(axi_stream &output_stream, float output[FC2_OUTPUT]) {
    for (int i = 0; i < FC2_OUTPUT; i++) {
        #pragma HLS PIPELINE
        axi_t temp;
        temp.data = output[i];
        output_stream.write(temp);
    }
    cout<<"Out"<<endl;
    for (int i = 0; i < FC2_OUTPUT; i++) {
    	cout << output[i] << endl;
    }
    float maxVal = output[0];
    int maxID = 0;
    for (int i = 1; i < FC2_OUTPUT; i++) {
    	if (output[i] > maxVal) {
    		maxVal = output[i];
    		maxID = i;
    	}
    }
    cout << "Maximum value of Dense Layer 2 is : " << maxVal << " for Class Id :" << maxID << endl;
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

// Convolution Layer 2
void conv2d_2(float conv1_output[c2_w_in][c2_l_in][c2_d_in], float conv2_output[c2_w_out][c2_l_out][c2_d_out]) {
    for (int f = 0; f < 32; f++) {
        for (int i = 0; i < 22; i++) {
            for (int j = 0; j < 22; j++) {
                #pragma HLS PIPELINE
                float sum = conv2_biases[f];
                for (int c = 0; c < 32; c++) {
                    for (int ki = 0; ki < 5; ki++) {
                        for (int kj = 0; kj < 5; kj++) {
                            sum += conv1_output[i + ki][j + kj][c] * conv2_weights[f][c][ki][kj];
                        }
                    }
                }
                conv2_output[i][j][f] = fmaxf(0.0f, sum); // ReLU Activation
            }
        }
    }
}

// Max Pooling Layer 1 (2x2)
void maxpool2d_1(float conv2_output[mp1_w_in][mp1_l_in][mp1_d_in], float maxpool1_output[mp1_w_out][mp1_l_out][mp1_d_out]) {
    for (int f = 0; f < 32; f++) {
        for (int i = 0; i < 11; i++) {
            for (int j = 0; j < 11; j++) {
                #pragma HLS PIPELINE
                float max_val = -FLT_MAX;
                for (int ki = 0; ki < 2; ki++) {
                    for (int kj = 0; kj < 2; kj++) {
                        max_val = fmaxf(max_val, conv2_output[2 * i + ki][2 * j + kj][f]);
                    }
                }
                maxpool1_output[i][j][f] = max_val;
            }
        }
    }
}

// Convolution Layer 3 (64 filters of size 3x3)
void conv2d_3(float maxpool1_output[c3_w_in][c3_l_in][c3_d_in], float conv3_output[c3_w_out][c3_l_out][c3_d_out]) {
    for (int f = 0; f < 64; f++) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                #pragma HLS PIPELINE
                float sum = conv3_biases[f];
                for (int c = 0; c < 32; c++) {
                    for (int ki = 0; ki < 3; ki++) {
                        for (int kj = 0; kj < 3; kj++) {
                            sum += maxpool1_output[i + ki][j + kj][c] * conv3_weights[f][c][ki][kj];
                        }
                    }
                }
                conv3_output[i][j][f] = fmaxf(0.0f, sum); // ReLU Activation
            }
        }
    }
}

// Convolution Layer 4 (64 filters of size 3x3)
void conv2d_4(float conv3_output[c4_w_in][c4_l_in][c4_d_in], float conv4_output[c4_w_out][c4_l_out][c4_d_out]) {
    for (int f = 0; f < 64; f++) {
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                #pragma HLS PIPELINE
                float sum = conv4_biases[f];
                for (int c = 0; c < 64; c++) {
                    for (int ki = 0; ki < 3; ki++) {
                        for (int kj = 0; kj < 3; kj++) {
                            sum += conv3_output[i + ki][j + kj][c] * conv4_weights[f][c][ki][kj];
                        }
                    }
                }
                conv4_output[i][j][f] = fmaxf(0.0f, sum); // ReLU Activation
            }
        }
    }
}

// Max Pooling Layer 2
void maxpool2d_2(float conv4_output[mp2_w_in][mp2_l_in][mp2_d_in], float maxpool2_output[mp2_w_out][mp2_l_out][mp2_d_out]) {
    for (int f = 0; f < 64; f++) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                #pragma HLS PIPELINE
                float max_val = -FLT_MAX;
                for (int ki = 0; ki < 2; ki++) {
                    for (int kj = 0; kj < 2; kj++) {
                        max_val = fmaxf(max_val, conv4_output[2 * i + ki][2 * j + kj][f]);
                    }
                }
                maxpool2_output[i][j][f] = max_val;
            }
        }
    }
}

// Flatten Layer
void flatten(float maxpool2_output[f_w_in][f_l_in][f_d_in], float flat_output[f_out]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int f = 0; f < 64; f++) {
                #pragma HLS PIPELINE
            	flat_output[i * 3 * 64 + j * 64 + f] = maxpool2_output[i][j][f];
            }
        }
    }
}

// Dense Layer 1
void dense_1(float flat_output[d1_in], float dense1_output[d1_out]) {
    for (int i = 0; i < 256; i++) {
        #pragma HLS PIPELINE
        float sum = fc1_biases[i];
        for (int j = 0; j < 576; j++) {
            sum += flat_output[j] * fc1_weights[i][j];
        }
        dense1_output[i] = fmaxf(0.0f, sum); // ReLU Activation
    }
}

// Dense Layer 2
void dense_2(float dense1_output[d2_in], float output[d2_out]) {
    for (int i = 0; i < 43; i++) {
        #pragma HLS PIPELINE
        float sum = fc2_biases[i];
        for (int j = 0; j < 256; j++) {
            sum += dense1_output[j] * fc2_weights[i][j];
        }
        output[i] = sum; // No activation, final output is logits
    }
}

// The CNN forward function, calling all layers in sequence
int cnn(axi_stream &input_stream, axi_stream &output_stream) {
#pragma HLS INTERFACE axis port=input_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    if (input_stream.empty()) {
        return -1; // Error: input stream is empty
    }

    float input[30][30][3];
    read_input(input_stream, input);

    float conv1_output[26][26][32];
    conv2d_1(input, conv1_output);

    float conv2_output[22][22][32];
    conv2d_2(conv1_output, conv2_output);

    float maxpool1_output[11][11][32];
    maxpool2d_1(conv2_output, maxpool1_output);

    float conv3_output[9][9][64];
    conv2d_3(maxpool1_output, conv3_output);

    float conv4_output[7][7][64];
    conv2d_4(conv3_output, conv4_output);

    float maxpool2_output[3][3][64];
    maxpool2d_2(conv4_output, maxpool2_output);

    float flat_output[576];
    flatten(maxpool2_output, flat_output);

    float dense1_output[256];
    dense_1(flat_output, dense1_output);

    float dense2_output[43];
    dense_2(dense1_output, dense2_output);

    write_output(output_stream, dense2_output);

    // Verify the correctness of the output stream
    if (output_stream.empty()) {
        return -2; // Error: output stream is empty
    }

    return 0; // Success
}
