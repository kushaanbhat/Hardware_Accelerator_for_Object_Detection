// ==============================================================
// Generated by Vitis HLS v2023.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="matrix_multiplier_matrix_multiplier,hls_ip_2023_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xc7z020-clg400-1,HLS_INPUT_CLOCK=10.000000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=7.256000,HLS_SYN_LAT=-1,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=1000,HLS_SYN_LUT=1497,HLS_VERSION=2023_2}" *)

module matrix_multiplier (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        A,
        x,
        y,
        y_ap_vld,
        m
);

parameter    ap_ST_fsm_state1 = 9'd1;
parameter    ap_ST_fsm_state2 = 9'd2;
parameter    ap_ST_fsm_state3 = 9'd4;
parameter    ap_ST_fsm_state4 = 9'd8;
parameter    ap_ST_fsm_state5 = 9'd16;
parameter    ap_ST_fsm_state6 = 9'd32;
parameter    ap_ST_fsm_state7 = 9'd64;
parameter    ap_ST_fsm_state8 = 9'd128;
parameter    ap_ST_fsm_state9 = 9'd256;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [31:0] A;
input  [31:0] x;
output  [31:0] y;
output   y_ap_vld;
input  [31:0] m;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg y_ap_vld;

(* fsm_encoding = "none" *) reg   [8:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [0:0] cmp2_fu_112_p2;
reg   [0:0] cmp2_reg_247;
wire   [31:0] bitcast_ln9_fu_118_p1;
reg   [31:0] bitcast_ln9_reg_252;
wire   [3:0] trunc_ln7_fu_141_p1;
reg   [3:0] trunc_ln7_reg_260;
wire    ap_CS_fsm_state2;
wire   [63:0] zext_ln17_fu_150_p1;
wire   [31:0] bitcast_ln17_fu_165_p1;
reg   [31:0] bitcast_ln17_reg_285;
wire    ap_CS_fsm_state5;
wire   [63:0] grp_fu_108_p2;
reg   [63:0] mul_ln17_reg_290;
wire   [0:0] icmp_ln14_fu_186_p2;
reg   [0:0] icmp_ln14_reg_301;
wire    ap_CS_fsm_state6;
wire   [3:0] trunc_ln14_fu_202_p1;
reg   [3:0] trunc_ln14_reg_306;
wire    ap_CS_fsm_state7;
reg   [3:0] x_local_address0;
reg    x_local_ce0;
reg    x_local_we0;
wire   [31:0] x_local_q0;
wire    grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start;
wire    grp_matrix_multiplier_Pipeline_L2_fu_91_ap_done;
wire    grp_matrix_multiplier_Pipeline_L2_fu_91_ap_idle;
wire    grp_matrix_multiplier_Pipeline_L2_fu_91_ap_ready;
wire   [3:0] grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_address0;
wire    grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_ce0;
wire    grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_we0;
wire   [31:0] grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_d0;
wire    grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start;
wire    grp_matrix_multiplier_Pipeline_L5_fu_99_ap_done;
wire    grp_matrix_multiplier_Pipeline_L5_fu_99_ap_idle;
wire    grp_matrix_multiplier_Pipeline_L5_fu_99_ap_ready;
wire   [3:0] grp_matrix_multiplier_Pipeline_L5_fu_99_x_local_address0;
wire    grp_matrix_multiplier_Pipeline_L5_fu_99_x_local_ce0;
wire   [31:0] grp_matrix_multiplier_Pipeline_L5_fu_99_y_tmp_out;
wire    grp_matrix_multiplier_Pipeline_L5_fu_99_y_tmp_out_ap_vld;
reg    grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start_reg;
wire    ap_CS_fsm_state3;
wire    ap_CS_fsm_state4;
reg    grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start_reg;
wire    ap_CS_fsm_state8;
reg   [31:0] i_fu_46;
wire   [31:0] add_ln7_fu_135_p2;
wire   [0:0] icmp_ln7_fu_130_p2;
wire    ap_CS_fsm_state9;
reg   [31:0] j_fu_58;
wire   [31:0] add_ln14_fu_207_p2;
wire   [0:0] icmp_ln13_fu_172_p2;
reg   [63:0] indvar_flatten_fu_62;
wire   [63:0] add_ln13_fu_177_p2;
wire   [31:0] grp_fu_108_p0;
wire   [31:0] grp_fu_108_p1;
wire   [31:0] select_ln13_fu_196_p3;
reg   [8:0] ap_NS_fsm;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
reg    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
wire    ap_ST_fsm_state6_blk;
wire    ap_ST_fsm_state7_blk;
reg    ap_ST_fsm_state8_blk;
wire    ap_ST_fsm_state9_blk;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 9'd1;
#0 grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start_reg = 1'b0;
#0 grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start_reg = 1'b0;
#0 i_fu_46 = 32'd0;
#0 j_fu_58 = 32'd0;
#0 indvar_flatten_fu_62 = 64'd0;
end

matrix_multiplier_x_local_RAM_AUTO_1R1W #(
    .DataWidth( 32 ),
    .AddressRange( 9 ),
    .AddressWidth( 4 ))
x_local_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(x_local_address0),
    .ce0(x_local_ce0),
    .we0(x_local_we0),
    .d0(grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_d0),
    .q0(x_local_q0)
);

matrix_multiplier_matrix_multiplier_Pipeline_L2 grp_matrix_multiplier_Pipeline_L2_fu_91(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start),
    .ap_done(grp_matrix_multiplier_Pipeline_L2_fu_91_ap_done),
    .ap_idle(grp_matrix_multiplier_Pipeline_L2_fu_91_ap_idle),
    .ap_ready(grp_matrix_multiplier_Pipeline_L2_fu_91_ap_ready),
    .empty(trunc_ln7_reg_260),
    .cmp2(cmp2_reg_247),
    .x_local_address0(grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_address0),
    .x_local_ce0(grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_ce0),
    .x_local_we0(grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_we0),
    .x_local_d0(grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_d0),
    .bitcast_ln9(bitcast_ln9_reg_252)
);

matrix_multiplier_matrix_multiplier_Pipeline_L5 grp_matrix_multiplier_Pipeline_L5_fu_99(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start),
    .ap_done(grp_matrix_multiplier_Pipeline_L5_fu_99_ap_done),
    .ap_idle(grp_matrix_multiplier_Pipeline_L5_fu_99_ap_idle),
    .ap_ready(grp_matrix_multiplier_Pipeline_L5_fu_99_ap_ready),
    .m(m),
    .empty(trunc_ln14_reg_306),
    .x_local_address0(grp_matrix_multiplier_Pipeline_L5_fu_99_x_local_address0),
    .x_local_ce0(grp_matrix_multiplier_Pipeline_L5_fu_99_x_local_ce0),
    .x_local_q0(x_local_q0),
    .bitcast_ln17(bitcast_ln17_reg_285),
    .y_tmp_out(grp_matrix_multiplier_Pipeline_L5_fu_99_y_tmp_out),
    .y_tmp_out_ap_vld(grp_matrix_multiplier_Pipeline_L5_fu_99_y_tmp_out_ap_vld)
);

matrix_multiplier_mul_32ns_32ns_64_2_1 #(
    .ID( 1 ),
    .NUM_STAGE( 2 ),
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .dout_WIDTH( 64 ))
mul_32ns_32ns_64_2_1_U14(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(grp_fu_108_p0),
    .din1(grp_fu_108_p1),
    .ce(1'b1),
    .dout(grp_fu_108_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state3)) begin
            grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start_reg <= 1'b1;
        end else if ((grp_matrix_multiplier_Pipeline_L2_fu_91_ap_ready == 1'b1)) begin
            grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_state7)) begin
            grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start_reg <= 1'b1;
        end else if ((grp_matrix_multiplier_Pipeline_L5_fu_99_ap_ready == 1'b1)) begin
            grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        i_fu_46 <= 32'd0;
    end else if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln7_fu_130_p2 == 1'd0))) begin
        i_fu_46 <= add_ln7_fu_135_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln7_fu_130_p2 == 1'd1))) begin
        indvar_flatten_fu_62 <= 64'd0;
    end else if (((1'b1 == ap_CS_fsm_state6) & (icmp_ln13_fu_172_p2 == 1'd0))) begin
        indvar_flatten_fu_62 <= add_ln13_fu_177_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln7_fu_130_p2 == 1'd1))) begin
        j_fu_58 <= 32'd0;
    end else if ((1'b1 == ap_CS_fsm_state7)) begin
        j_fu_58 <= add_ln14_fu_207_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        bitcast_ln17_reg_285 <= bitcast_ln17_fu_165_p1;
        mul_ln17_reg_290 <= grp_fu_108_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state1)) begin
        bitcast_ln9_reg_252 <= bitcast_ln9_fu_118_p1;
        cmp2_reg_247 <= cmp2_fu_112_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        icmp_ln14_reg_301 <= icmp_ln14_fu_186_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        trunc_ln14_reg_306 <= trunc_ln14_fu_202_p1;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        trunc_ln7_reg_260 <= trunc_ln7_fu_141_p1;
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

always @ (*) begin
    if ((grp_matrix_multiplier_Pipeline_L2_fu_91_ap_done == 1'b0)) begin
        ap_ST_fsm_state4_blk = 1'b1;
    end else begin
        ap_ST_fsm_state4_blk = 1'b0;
    end
end

assign ap_ST_fsm_state5_blk = 1'b0;

assign ap_ST_fsm_state6_blk = 1'b0;

assign ap_ST_fsm_state7_blk = 1'b0;

always @ (*) begin
    if ((grp_matrix_multiplier_Pipeline_L5_fu_99_ap_done == 1'b0)) begin
        ap_ST_fsm_state8_blk = 1'b1;
    end else begin
        ap_ST_fsm_state8_blk = 1'b0;
    end
end

assign ap_ST_fsm_state9_blk = 1'b0;

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (icmp_ln13_fu_172_p2 == 1'd1))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (icmp_ln13_fu_172_p2 == 1'd1))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        x_local_address0 = grp_matrix_multiplier_Pipeline_L5_fu_99_x_local_address0;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        x_local_address0 = grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_address0;
    end else begin
        x_local_address0 = 'bx;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        x_local_ce0 = grp_matrix_multiplier_Pipeline_L5_fu_99_x_local_ce0;
    end else if ((1'b1 == ap_CS_fsm_state4)) begin
        x_local_ce0 = grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_ce0;
    end else begin
        x_local_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        x_local_we0 = grp_matrix_multiplier_Pipeline_L2_fu_91_x_local_we0;
    end else begin
        x_local_we0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        y_ap_vld = 1'b1;
    end else begin
        y_ap_vld = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((1'b1 == ap_CS_fsm_state2) & (icmp_ln7_fu_130_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            if (((grp_matrix_multiplier_Pipeline_L2_fu_91_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            if (((1'b1 == ap_CS_fsm_state6) & (icmp_ln13_fu_172_p2 == 1'd1))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            if (((grp_matrix_multiplier_Pipeline_L5_fu_99_ap_done == 1'b1) & (1'b1 == ap_CS_fsm_state8))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign add_ln13_fu_177_p2 = (indvar_flatten_fu_62 + 64'd1);

assign add_ln14_fu_207_p2 = (select_ln13_fu_196_p3 + 32'd1);

assign add_ln7_fu_135_p2 = (i_fu_46 + 32'd1);

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign bitcast_ln17_fu_165_p1 = A;

assign bitcast_ln9_fu_118_p1 = x;

assign cmp2_fu_112_p2 = ((m == 32'd0) ? 1'b1 : 1'b0);

assign grp_fu_108_p0 = zext_ln17_fu_150_p1;

assign grp_fu_108_p1 = zext_ln17_fu_150_p1;

assign grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start = grp_matrix_multiplier_Pipeline_L2_fu_91_ap_start_reg;

assign grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start = grp_matrix_multiplier_Pipeline_L5_fu_99_ap_start_reg;

assign icmp_ln13_fu_172_p2 = ((indvar_flatten_fu_62 == mul_ln17_reg_290) ? 1'b1 : 1'b0);

assign icmp_ln14_fu_186_p2 = ((j_fu_58 == m) ? 1'b1 : 1'b0);

assign icmp_ln7_fu_130_p2 = ((i_fu_46 == m) ? 1'b1 : 1'b0);

assign select_ln13_fu_196_p3 = ((icmp_ln14_reg_301[0:0] == 1'b1) ? 32'd0 : j_fu_58);

assign trunc_ln14_fu_202_p1 = select_ln13_fu_196_p3[3:0];

assign trunc_ln7_fu_141_p1 = i_fu_46[3:0];

assign y = grp_matrix_multiplier_Pipeline_L5_fu_99_y_tmp_out;

assign zext_ln17_fu_150_p1 = m;

endmodule //matrix_multiplier
