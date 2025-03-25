#ifndef _CNN_H_
#define _CNN_H_

<<<<<<< HEAD
#include "ap_axi_sdata.h"
=======
// Define AXI Stream types
typedef ap_fixed<16,6> fixed_t;
typedef hls::stream<fixed_t> axi_stream;
>>>>>>> 230aa7bf74dbbd213528fd1a18ff58b0c6c4fdf9

#define img_h 30
#define img_w 30
#define img_d 3

#define w_1_h 3
#define w_1_w 3
#define w_1_d 32

#define c_out_h 28
#define c_out_w 28
#define c_out_d 32

#define FLT_MAX 3.402823466e+38F

#define m_out_h 14
#define m_out_w 14
#define m_out_d 32

#define f_out_h 6272

#define d_out_h 43


// Define AXI interface type (32-bit float data)
typedef ap_axiu<32, 0, 0, 0> AXI_VAL;

// AXI Interface to Floating-point conversion helper function
inline float axi_to_float(AXI_VAL &input) {
    union { unsigned int i; float f; } converter;
    converter.i = input.data;
    return converter.f;
}

// Floating-point to AXI Interface conversion helper function
inline AXI_VAL float_to_axi(float val) {
    AXI_VAL output;
    union { unsigned int i; float f; } converter;
    converter.f = val;
    output.data = converter.i;
    output.last = 0;
    return output;
}

void conv2d( float image[img_h][img_w][img_d],float c_out_1[c_out_h][c_out_w][c_out_d]);
void maxpool2d(float c_out[c_out_h][c_out_w][c_out_d], float m_out[m_out_h][m_out_w][m_out_d]);
void flatten(float m_out[m_out_h][m_out_w][m_out_d], float f_out[f_out_h]);
void dense(float f_out[f_out_h], float d_out[d_out_h]);

#endif
