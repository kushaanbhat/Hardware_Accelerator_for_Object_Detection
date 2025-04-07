#ifndef _CNN_H_
#define _CNN_H_

#include "ap_axi_sdata.h"

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
typedef ap_axiu<32, 0, 0, 0> AXI_VAL;// 32-bit AXI stream type
typedef ap_fixed<16, 16> data;   // 16-bit fixed-point with 8 integer bits

void conv2d( data image[img_h][img_w][img_d], data c_out_1[c_out_h][c_out_w][c_out_d]);
void maxpool2d(data c_out[c_out_h][c_out_w][c_out_d], data m_out[m_out_h][m_out_w][m_out_d]);
void flatten( data m_out[m_out_h][m_out_w][m_out_d], data f_out[f_out_h]);
void dense( data f_out[f_out_h], data d_out[d_out_h]);

#endif
