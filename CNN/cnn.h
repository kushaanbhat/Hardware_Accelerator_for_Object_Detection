#ifndef CNN_H
#define CNN_H
#include "iostream"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>

// Define AXI Stream types
typedef ap_fixed<16,6> fixed_t;
typedef hls::stream<fixed_t> axi_stream;

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

#define c1_w_in INPUT_WIDTH
#define c1_l_in INPUT_HEIGHT
#define c1_d_in INPUT_CHANNELS

#define c1_w_out (INPUT_WIDTH-KERNEL_SIZE_5x5+1)
#define c1_l_out (INPUT_HEIGHT-KERNEL_SIZE_5x5+1)
#define c1_d_out CONV1_FILTERS


#define c2_w_in (INPUT_WIDTH-KERNEL_SIZE_5x5+1)
#define c2_l_in (INPUT_HEIGHT-KERNEL_SIZE_5x5+1)
#define c2_d_in CONV1_FILTERS

#define c2_w_out ((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)
#define c2_l_out ((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)
#define c2_d_out CONV2_FILTERS


#define mp1_w_in ((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)
#define mp1_l_in ((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)
#define mp1_d_in CONV2_FILTERS

#define mp1_w_out (((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)
#define mp1_l_out (((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)
#define mp1_d_out CONV2_FILTERS


#define c3_w_in (((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)
#define c3_l_in (((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)
#define c3_d_in CONV2_FILTERS

#define c3_w_out ((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)
#define c3_l_out ((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)
#define c3_d_out CONV3_FILTERS


#define c4_w_in ((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)
#define c4_l_in ((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)
#define c4_d_in CONV3_FILTERS

#define c4_w_out (((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)
#define c4_l_out (((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)
#define c4_d_out CONV4_FILTERS


#define mp2_w_in (((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)
#define mp2_l_in (((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)
#define mp2_d_in CONV4_FILTERS

#define mp2_w_out ((((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)
#define mp2_l_out ((((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)
#define mp2_d_out CONV4_FILTERS


#define f_w_in ((((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)
#define f_l_in ((((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)
#define f_d_in CONV4_FILTERS

#define f_out ((((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)*((((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)*CONV4_FILTERS

#define d1_in ((((((INPUT_WIDTH-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)*((((((INPUT_HEIGHT-KERNEL_SIZE_5x5+1)-KERNEL_SIZE_5x5+1)/2)-KERNEL_SIZE_3x3+1)-KERNEL_SIZE_3x3+1)/2)*CONV4_FILTERS

#define d1_out FC1_OUTPUT


#define d2_in FC1_OUTPUT

#define d2_out FC2_OUTPUT

// Function Prototypes
void cnn(axi_stream &input_stream, axi_stream &output_stream);

// Layer functions
void read_input(axi_stream &input_stream, float input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS]);
void write_output(axi_stream &output_stream, float output[FC2_OUTPUT]);
void conv2d_1(float input[c1_w_in][c1_l_in][c1_d_in], float output[c1_w_out][c1_l_out][c1_d_out]);
void conv2d_2(float input[c2_w_in][c2_l_in][c2_d_in], float output[c2_w_out][c2_l_out][c2_d_out]);
void maxpool2d_1(float input[mp1_w_in][mp1_l_in][mp1_d_in], float output[mp1_w_out][mp1_l_out][mp1_d_out]);
void conv2d_3(float input[c3_w_in][c3_l_in][c3_d_in], float output[c3_w_out][c3_l_out][c3_d_out]);
void conv2d_4(float input[c4_w_in][c4_l_in][c4_d_in], float output[c4_w_out][c4_l_out][c4_d_out]);
void maxpool2d_2(float input[mp2_w_in][mp2_l_in][mp2_d_in], float output[mp2_w_out][mp2_l_out][mp2_d_out]);
void flatten(float input[f_w_in][f_l_in][f_d_in], float output[f_out]);
void dense_1(float input[d1_in], float output[d1_out]);
void dense_2(float input[d2_in], float output[d2_out]);

#endif
