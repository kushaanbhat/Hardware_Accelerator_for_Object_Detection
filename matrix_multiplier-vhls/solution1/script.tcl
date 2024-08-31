############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
## Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
############################################################
open_project matrix_multiplier-vhls
set_top matrix_multiplier
add_files matrix_multiplier-vhls/matrix_multiplier.cpp
add_files -tb matrix_multiplier-vhls/matrix_multiplier_tb.cpp -cflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xc7z020-clg400-1}
create_clock -period 10 -name default
source "./matrix_multiplier-vhls/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -format ip_catalog
