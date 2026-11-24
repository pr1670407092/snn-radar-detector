`timescale 1ns/1ps
// 2D convolution + LIF + SEW engine, resource-balanced build.
//
// 128 lanes, one tap/cycle.  A 192-word x 256-bit FF weight cache removes
// the weight-BRAM read from the MAC loop; L5/L6's remaining 96 words live in
// a small distributed-RAM tail.  The cache shifts one word per tap so its
// output is always wcache[0] (no 192:1 read mux).
module conv2d_engine (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         busy,
    output reg         done,

    input  wire [5:0]  cfg_cin,
    input  wire [5:0]  cfg_cout,
    input  wire [5:0]  cfg_hin,
    input  wire [5:0]  cfg_win,
    input  wire [5:0]  cfg_hout,
    input  wire [5:0]  cfg_wout,
    input  wire [1:0]  cfg_stride,
    input  wire        cfg_sew,
    input  wire [10:0] cfg_wbase,
    input  wire [7:0]  cfg_bbase,
    input  wire [9:0]  cfg_vbase,
    input  wire [11:0] cfg_spkin_base,
    input  wire [11:0] cfg_spkout_base,
    input  wire [11:0] cfg_sew_base,
    input  wire signed [23:0] cfg_th,

    output reg  [10:0] w_addr,
    input  wire [255:0] w_data,
    output reg  [7:0]  b_addr,
    input  wire [23:0] b_data,

    output reg  [11:0] spk_raddr,
    input  wire [31:0] spk_rdata,
    output reg         spk_we,
    output reg  [11:0] spk_waddr,
    output reg  [31:0] spk_wdata,

    output reg  [9:0]  v_raddr,
    input  wire [3071:0] v_rdata,
    output reg         v_we,
    output reg  [9:0]  v_waddr,
    output reg  [3071:0] v_wdata,

    output reg  [17:0] spk_count
);
    localparam S_IDLE      = 5'd0,
               S_BIAS_SET  = 5'd1,
               S_BIAS_WAIT = 5'd2,
               S_BIAS_GET  = 5'd3,
               S_WC_LOAD   = 5'd4,
               S_TL_SET    = 5'd5,
               S_TL_WAIT   = 5'd6,
               S_TL_GET    = 5'd7,
               S_LD_SET    = 5'd8,
               S_LD_WAIT   = 5'd9,
               S_LD_GET    = 5'd10,
               S_GRP_INIT  = 5'd11,
               S_GRP_WAIT  = 5'd12,
               S_MAC       = 5'd13,
               S_RES_SET   = 5'd14,
               S_RES_WAIT  = 5'd15,
               S_RES_GET   = 5'd16,
               S_LIF       = 5'd17,
               S_NEXT_GRP  = 5'd18,
               S_ROW_WR    = 5'd19,
               S_NEXT_ROW  = 5'd20,
               S_CNT       = 5'd21,
               S_LIF2      = 5'd22,
               S_WC_WAIT   = 5'd23,
               S_WC_GET    = 5'd24;

    reg [4:0]  state;
    reg [5:0]  row;
    reg [2:0]  group;
    reg [1:0]  ld_ky;
    reg [5:0]  ld_ci;
    reg        ld_zero;
    reg [5:0]  b_idx;
    reg [5:0]  res_idx;
    reg [5:0]  wr_co;
    reg [5:0]  base_i;

    (* ram_style = "registers", shreg_extract = "no", srl_style = "register" *)
    reg [255:0] wcache [0:191];
    (* ram_style = "distributed" *) reg [255:0] tail_mem [0:95];
    reg [7:0]  load_cnt;
    reg [8:0]  p0, ld_p;
    reg [6:0]  tail_i;
    reg [8:0]  tap_i;
    reg [7:0]  sg_r;
    reg [5:0]  c_ci;
    reg [1:0]  c_ky, c_kx;

    reg signed [23:0] acc [0:127];
    reg signed [23:0] bias [0:31];
    reg               resvec [0:127];
    reg [33:0]        lbuf [0:2][0:31];
    reg [31:0]        obuf [0:31];

    integer l, j, co, g, row_i;

    wire [8:0] tap_words = {3'd0, cfg_cin} * 9;
    wire [8:0] pat_len = (cfg_cin == 6'd1) ? 9'd16 : 9'd192;
    wire [8:0] proc_len = (tap_words > 9'd192) ? 9'd288 : pat_len;
    wire [8:0] tail_idx = (tap_i >= 9'd192) ? (tap_i - 9'd192) : 9'd0;
    wire [255:0] w_cur = (tap_i < 9'd192) ? wcache[0] : tail_mem[tail_idx];

    wire signed [25:0] th_ext = {{2{cfg_th[23]}}, cfg_th};
    reg signed [25:0] v_prev, cur, v_sum, v_new;
    reg               fire;
    reg [3071:0]      v_pack;
    reg [7:0]         pc;
    reg [7:0]         pc_q;
    reg [127:0]       fire_vec;

    always @* begin
        base_i = group * ((cfg_cout == 6'd16) ? 6'd8 : 6'd4) * cfg_stride;
    end

    function [7:0] win_coord;
        input [5:0] fci;
        input [1:0] fky, fkx;
        integer jj;
        begin
            win_coord = 8'd0;
            for (jj = 0; jj < 8; jj = jj + 1)
                if (jj < ((cfg_cout == 6'd16) ? 8 : 4))
                    win_coord[jj] = lbuf[fky][fci][base_i + jj * cfg_stride + fkx];
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            spk_we <= 1'b0;
            v_we <= 1'b0;
            spk_count <= 18'd0;
            w_addr <= 11'd0;
            b_addr <= 8'd0;
            spk_raddr <= 12'd0;
            spk_waddr <= 12'd0;
            spk_wdata <= 32'd0;
            v_raddr <= 10'd0;
            v_waddr <= 10'd0;
            v_wdata <= 3072'd0;
            row <= 6'd0;
            group <= 3'd0;
            ld_ky <= 2'd0;
            ld_ci <= 6'd0;
            ld_zero <= 1'b0;
            b_idx <= 6'd0;
            res_idx <= 6'd0;
            wr_co <= 6'd0;
            load_cnt <= 8'd0;
            p0 <= 9'd0;
            ld_p <= 9'd0;
            tail_i <= 7'd0;
            tap_i <= 9'd0;
            c_ci <= 6'd0;
            c_ky <= 2'd0;
            c_kx <= 2'd0;
            sg_r <= 8'd0;
            pc_q <= 8'd0;
            for (l = 0; l < 192; l = l + 1)
                wcache[l] <= 256'd0;
            for (l = 0; l < 128; l = l + 1) begin
                acc[l] <= 24'sd0;
                resvec[l] <= 1'b0;
            end
        end else begin
            spk_we <= 1'b0;
            v_we   <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        done <= 1'b0;
                        spk_count <= 18'd0;
                        b_idx <= 6'd0;
                        state <= S_BIAS_SET;
                    end
                end

                S_BIAS_SET: begin
                    b_addr <= cfg_bbase + {2'b0, b_idx};
                    state <= S_BIAS_WAIT;
                end
                S_BIAS_WAIT: state <= S_BIAS_GET;
                S_BIAS_GET: begin
                    bias[b_idx[4:0]] <= $signed(b_data);
                    if (b_idx == 6'd31) begin
                        load_cnt <= 8'd0;
                        p0 <= 9'd0;
                        state <= S_WC_LOAD;
                    end else begin
                        b_idx <= b_idx + 6'd1;
                        state <= S_BIAS_SET;
                    end
                end

                // fill 192 cache words (3-cycle ROM read per word)
                S_WC_LOAD: begin
                    w_addr <= cfg_wbase + {2'd0, p0};
                    state <= S_WC_WAIT;
                end
                S_WC_WAIT: state <= S_WC_GET;
                S_WC_GET: begin
                    for (l = 0; l < 191; l = l + 1)
                        wcache[l] <= wcache[l + 1];
                    wcache[191] <= (p0 < tap_words) ? w_data : 256'd0;
                    if (p0 + 9'd1 >= pat_len) p0 <= 9'd0;
                    else                      p0 <= p0 + 9'd1;
                    if (load_cnt == 8'd191) begin
                        load_cnt <= 8'd0;
                        if (tap_words > 9'd192) begin
                            tail_i <= 7'd0;
                            state <= S_TL_SET;
                        end else begin
                            row <= 6'd0;
                            group <= 3'd0;
                            ld_ky <= 2'd0;
                            ld_ci <= 6'd0;
                            state <= S_LD_SET;
                        end
                    end else begin
                        load_cnt <= load_cnt + 8'd1;
                        state <= S_WC_LOAD;
                    end
                end

                // 96 tail words for L5/L6
                S_TL_SET: begin
                    w_addr <= cfg_wbase + 9'd192 + {2'd0, tail_i};
                    state <= S_TL_WAIT;
                end
                S_TL_WAIT: state <= S_TL_GET;
                S_TL_GET: begin
                    tail_mem[tail_i] <= w_data;
                    if (tail_i == 7'd95) begin
                        row <= 6'd0;
                        group <= 3'd0;
                        ld_ky <= 2'd0;
                        ld_ci <= 6'd0;
                        state <= S_LD_SET;
                    end else begin
                        tail_i <= tail_i + 7'd1;
                        state <= S_TL_SET;
                    end
                end

                S_LD_SET: begin
                    row_i = row * cfg_stride + ld_ky - 1;
                    if (row_i < 0 || row_i >= cfg_hin) begin
                        ld_zero <= 1'b1;
                        spk_raddr <= 12'd0;
                    end else begin
                        ld_zero <= 1'b0;
                        spk_raddr <= cfg_spkin_base + ld_ci * cfg_hin + row_i;
                    end
                    state <= S_LD_WAIT;
                end
                S_LD_WAIT: state <= S_LD_GET;
                S_LD_GET: begin
                    if (ld_zero)
                        lbuf[ld_ky][ld_ci] <= 34'd0;
                    else
                        lbuf[ld_ky][ld_ci] <= {1'b0, spk_rdata, 1'b0};
                    if (ld_ci == cfg_cin - 1) begin
                        ld_ci <= 6'd0;
                        if (ld_ky == 2'd2) begin
                            ld_ky <= 2'd0;
                            state <= S_GRP_INIT;
                        end else begin
                            ld_ky <= ld_ky + 2'd1;
                            state <= S_LD_SET;
                        end
                    end else begin
                        ld_ci <= ld_ci + 6'd1;
                        state <= S_LD_SET;
                    end
                end

                S_GRP_INIT: begin
                    for (l = 0; l < 128; l = l + 1) begin
                        acc[l] <= 24'sd0;
                        resvec[l] <= 1'b0;
                    end
                    if (group == 3'd0)
                        for (l = 0; l < 32; l = l + 1)
                            obuf[l] <= 32'd0;
                    tap_i <= 9'd0;
                    v_raddr <= cfg_vbase + row * 4 + group;
                    state <= S_GRP_WAIT;
                end
                S_GRP_WAIT: begin
                    sg_r <= win_coord(6'd0, 2'd0, 2'd0);
                    c_ci <= 6'd0; c_ky <= 2'd0; c_kx <= 2'd1;
                    state <= S_MAC;
                end

                S_MAC: begin
                    for (l = 0; l < 128; l = l + 1) begin
                        co = (cfg_cout == 6'd16) ? (l & 15) : (l & 31);
                        g  = (cfg_cout == 6'd16) ? (l >> 4) : (l >> 5);
                        if (sg_r[g])
                            acc[l] <= acc[l] + $signed(w_cur[8*co +: 8]);
                    end
                    if (tap_i + 9'd1 < tap_words) begin
                        sg_r <= win_coord(c_ci, c_ky, c_kx);
                        if (c_kx == 2'd2) begin
                            c_kx <= 2'd0;
                            if (c_ky == 2'd2) begin
                                c_ky <= 2'd0;
                                c_ci <= c_ci + 6'd1;
                            end else c_ky <= c_ky + 2'd1;
                        end else c_kx <= c_kx + 2'd1;
                    end else begin
                        sg_r <= 8'd0;
                    end
                    if (tap_i < 9'd192) begin
                        for (l = 0; l < 191; l = l + 1)
                            wcache[l] <= wcache[l + 1];
                        wcache[191] <= wcache[0];
                    end
                    if (tap_i + 9'd1 >= proc_len) begin
                        if (cfg_sew) begin
                            res_idx <= 6'd0;
                            state <= S_RES_SET;
                        end else state <= S_LIF;
                    end else begin
                        tap_i <= tap_i + 9'd1;
                    end
                end

                S_RES_SET: begin
                    spk_raddr <= cfg_sew_base + res_idx * cfg_hout + row;
                    state <= S_RES_WAIT;
                end
                S_RES_WAIT: state <= S_RES_GET;
                S_RES_GET: begin
                    if (cfg_cout == 6'd16)
                        for (j = 0; j < 8; j = j + 1)
                            resvec[res_idx[4:0]*8 + j] <= spk_rdata[group*8 + j];
                    else
                        for (j = 0; j < 4; j = j + 1)
                            resvec[res_idx[4:0]*4 + j] <= spk_rdata[group*4 + j];
                    if (res_idx == cfg_cout - 1)
                        state <= S_LIF;
                    else begin
                        res_idx <= res_idx + 6'd1;
                        state <= S_RES_SET;
                    end
                end

                S_LIF: begin
                    v_pack = 3072'd0;
                    for (l = 0; l < 128; l = l + 1) begin
                        co = (cfg_cout == 6'd16) ? (l & 15) : (l & 31);
                        g  = (cfg_cout == 6'd16) ? (l >> 4) : (l >> 5);
                        v_prev = {{2{v_rdata[24*l + 23]}}, v_rdata[24*l +: 24]};
                        cur = {{2{acc[l][23]}}, acc[l]} +
                              {{2{bias[co][23]}}, bias[co]} +
                              (cfg_sew && resvec[co * ((cfg_cout == 6'd16) ? 8 : 4) + g]
                               ? {{2{cfg_th[23]}}, cfg_th} : 26'sd0);
                        v_sum = v_prev + cur;
                        fire = (v_sum >= th_ext);
                        v_new = fire ? (v_sum - th_ext) : v_sum;
                        fire_vec[l] <= fire;
                        v_pack[24*l +: 24] = v_new[23:0];
                    end
                    state <= S_LIF2;
                end

                S_LIF2: begin
                    pc = 8'd0;
                    for (l = 0; l < 128; l = l + 1) begin
                        co = (cfg_cout == 6'd16) ? (l & 15) : (l & 31);
                        g  = (cfg_cout == 6'd16) ? (l >> 4) : (l >> 5);
                        if (fire_vec[l]) begin
                            pc = pc + 8'd1;
                            obuf[co][group * ((cfg_cout == 6'd16) ? 8 : 4) + g] <= 1'b1;
                        end
                    end
                    pc_q <= pc;
                    v_we <= 1'b1;
                    v_waddr <= cfg_vbase + row * 4 + group;
                    v_wdata <= v_pack;
                    state <= S_CNT;
                end

                S_CNT: begin
                    spk_count <= spk_count + {10'd0, pc_q};
                    state <= S_NEXT_GRP;
                end

                S_NEXT_GRP: begin
                    if (group == 3'd3) begin
                        wr_co <= 6'd0;
                        state <= S_ROW_WR;
                    end else begin
                        group <= group + 3'd1;
                        state <= S_GRP_INIT;
                    end
                end

                S_ROW_WR: begin
                    spk_we <= 1'b1;
                    spk_waddr <= cfg_spkout_base + wr_co * cfg_hout + row;
                    spk_wdata <= obuf[wr_co[4:0]];
                    if (wr_co == cfg_cout - 1)
                        state <= S_NEXT_ROW;
                    else
                        wr_co <= wr_co + 6'd1;
                end

                S_NEXT_ROW: begin
                    for (l = 0; l < 32; l = l + 1)
                        obuf[l] <= 32'd0;
                    if (row == cfg_hout - 1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        row <= row + 6'd1;
                        group <= 3'd0;
                        ld_ky <= 2'd0;
                        ld_ci <= 6'd0;
                        state <= S_LD_SET;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
