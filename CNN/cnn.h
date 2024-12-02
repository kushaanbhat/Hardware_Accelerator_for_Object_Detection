#ifndef CNN_H
#define CNN_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

// Define AXI Stream types
typedef ap_axis<32, 2, 5, 6> axi_t;
typedef hls::stream<axi_t> axi_stream;

// CNN Layer Specifications
#define INPUT_WIDTH 30
#define INPUT_HEIGHT 30
#define INPUT_CHANNELS 3

#define CONV1_FILTERS 32
#define CONV2_FILTERS 32
#define CONV3_FILTERS 64
#define CONV4_FILTERS 64

#define KERNEL_SIZE_5x5 5
#define KERNEL_SIZE_3x3 3
#define POOL_SIZE 2

#define FC1_OUTPUT 256
#define FC2_OUTPUT 43

// Function Prototypes
void cnn(axi_stream &input_stream, axi_stream &output_stream);

// Layer functions
void read_input(axi_stream &input_stream, float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS]);
void write_output(axi_stream &output_stream, float output[FC2_OUTPUT]);

void conv2d_1(float input[30][30][3], float output[26][26][32]);
void conv2d_2(float input[26][26][32], float output[22][22][32]);
void maxpool2d_1(float input[22][22][32], float output[11][11][32]);
void conv2d_3(float input[11][11][32], float output[9][9][64]);
void conv2d_4(float input[9][9][64], float output[7][7][64]);
void maxpool2d_2(float input[7][7][64], float output[3][3][64]);
void flatten(float input[3][3][64], float output[576]);
void dense_1(float input[576], float output[256]);
void dense_2(float input[256], float output[43]);

#endif
