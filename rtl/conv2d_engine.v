`timescale 1ns/1ps
// 2D convolution + LIF + SEW engine for the time-frequency RSN.
//
// One engine handles all six layers; the top supplies a runtime layer config.
// 64 lanes = 4 pixels x 16 output channels (Cout=16) or 2 pixels x 32
// output channels (Cout=32).  One tap (Cin*3*3 taps total) is accumulated per
// cycle; the 3x3 window comes from a 3-row line buffer loaded from the global
// spike memory.  Weights are 8-bit, membrane is 24-bit signed, and the SEW
// residual adds one threshold unit.
module conv2d_engine (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    output reg         busy,
    output reg         done,

    // runtime layer config
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
    input  wire [10:0] cfg_vbase,
    input  wire [11:0] cfg_spkin_base,
    input  wire [11:0] cfg_spkout_base,
    input  wire [11:0] cfg_sew_base,
    input  wire signed [23:0] cfg_th,

    // weight / bias ROM ports (1-cycle registered data)
    output reg  [10:0] w_addr,
    input  wire [255:0] w_data,
    output reg  [7:0]  b_addr,
    input  wire [23:0] b_data,

    // global spike memory (1 read + 1 write port)
    output reg  [11:0] spk_raddr,
    input  wire [31:0] spk_rdata,
    output reg         spk_we,
    output reg  [11:0] spk_waddr,
    output reg  [31:0] spk_wdata,

    // wide membrane memory, 64 x 24-bit per address
    output reg  [10:0] v_raddr,
    input  wire [1535:0] v_rdata,
    output reg         v_we,
    output reg  [10:0] v_waddr,
    output reg  [1535:0] v_wdata,

    // number of output spikes in this layer (across all groups)
    output reg  [16:0] spk_count
);
    localparam S_IDLE      = 5'd0,
               S_BIAS_SET  = 5'd1,
               S_BIAS_WAIT = 5'd2,
               S_BIAS_GET  = 5'd3,
               S_LD_SET    = 5'd4,
               S_LD_WAIT   = 5'd5,
               S_LD_GET    = 5'd6,
               S_GRP_INIT  = 5'd7,
               S_GRP_WAIT  = 5'd8,
               S_MAC       = 5'd9,
               S_RES_SET   = 5'd10,
               S_RES_WAIT  = 5'd11,
               S_RES_GET   = 5'd12,
               S_LIF       = 5'd13,
               S_NEXT_GRP  = 5'd14,
               S_ROW_WR    = 5'd15,
               S_NEXT_ROW  = 6'd16;

    reg [4:0]  state;
    reg [5:0]  row;
    reg [2:0]  group;
    reg [8:0]  tap_d;
    reg [1:0]  ld_ky;
    reg [5:0]  ld_ci;
    reg        ld_zero;
    reg [5:0]  b_idx;
    reg [5:0]  res_idx;
    reg [5:0]  wr_co;
    reg [4:0]  ci_i;
    reg [1:0]  ky_i, kx_i;
    reg [5:0]  base_i;
    reg [3:0]  sg;

    reg signed [23:0] acc  [0:63];
    reg signed [23:0] bias [0:31];
    reg               resvec [0:63];
    reg [33:0]        lbuf [0:2][0:31];
    reg [31:0]        obuf [0:31];

    integer l, j, co, g, row_i;
    wire signed [25:0] th_ext = {{2{cfg_th[23]}}, cfg_th};
    reg signed [25:0] v_prev, cur, v_sum, v_new;
    reg               fire;
    reg [1535:0]      v_pack;
    reg [6:0]         pc;

    // decode the current tap / group window
    always @* begin
        ci_i  = tap_d / 9;
        ky_i  = (tap_d % 9) / 3;
        kx_i  = tap_d % 3;
        base_i = group * ((cfg_cout == 6'd16) ? 6'd4 : 6'd2) * cfg_stride;
        sg = 4'b0000;
        for (j = 0; j < 4; j = j + 1)
            sg[j] = lbuf[ky_i][ci_i][base_i + j * cfg_stride + kx_i];
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            spk_we <= 1'b0;
            v_we <= 1'b0;
            spk_count <= 17'd0;
            w_addr <= 11'd0;
            b_addr <= 8'd0;
            spk_raddr <= 12'd0;
            spk_waddr <= 12'd0;
            spk_wdata <= 32'd0;
            v_raddr <= 11'd0;
            v_waddr <= 11'd0;
            v_wdata <= 1536'd0;
            row <= 6'd0;
            group <= 3'd0;
            tap_d <= 9'd0;
            ld_ky <= 2'd0;
            ld_ci <= 6'd0;
            ld_zero <= 1'b0;
            b_idx <= 6'd0;
            res_idx <= 6'd0;
            wr_co <= 6'd0;
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
                        spk_count <= 17'd0;
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
                        row <= 6'd0;
                        group <= 3'd0;
                        ld_ky <= 2'd0;
                        ld_ci <= 6'd0;
                        state <= S_LD_SET;
                    end else begin
                        b_idx <= b_idx + 6'd1;
                        state <= S_BIAS_SET;
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
                    for (l = 0; l < 64; l = l + 1) begin
                        acc[l] <= 24'sd0;
                        resvec[l] <= 1'b0;
                    end
                    if (group == 3'd0)
                        for (l = 0; l < 32; l = l + 1)
                            obuf[l] <= 32'd0;
                    tap_d <= 9'd0;
                    w_addr <= cfg_wbase;
                    v_raddr <= cfg_vbase + row * 8 + group;
                    state <= S_GRP_WAIT;
                end
                S_GRP_WAIT: begin
                    w_addr <= cfg_wbase + 11'd1;
                    state <= S_MAC;
                end

                S_MAC: begin
                    for (l = 0; l < 64; l = l + 1) begin
                        co = (cfg_cout == 6'd16) ? (l & 15) : (l & 31);
                        g  = (cfg_cout == 6'd16) ? (l >> 4) : (l >> 5);
                        if (sg[g])
                            acc[l] <= acc[l] + $signed(w_data[8*co +: 8]);
                    end
                    if (tap_d == cfg_cin * 9 - 1) begin
                        if (cfg_sew) begin
                            res_idx <= 6'd0;
                            state <= S_RES_SET;
                        end else begin
                            state <= S_LIF;
                        end
                    end else begin
                        tap_d <= tap_d + 9'd1;
                        w_addr <= cfg_wbase + tap_d + 11'd2;
                    end
                end

                S_RES_SET: begin
                    spk_raddr <= cfg_sew_base + res_idx * cfg_hout + row;
                    state <= S_RES_WAIT;
                end
                S_RES_WAIT: state <= S_RES_GET;
                S_RES_GET: begin
                    if (cfg_cout == 6'd16)
                        for (j = 0; j < 4; j = j + 1)
                            resvec[res_idx[4:0]*4 + j] <= spk_rdata[group*4 + j];
                    else
                        for (j = 0; j < 2; j = j + 1)
                            resvec[res_idx[4:0]*2 + j] <= spk_rdata[group*2 + j];
                    if (res_idx == cfg_cout - 1)
                        state <= S_LIF;
                    else begin
                        res_idx <= res_idx + 6'd1;
                        state <= S_RES_SET;
                    end
                end

                S_LIF: begin
                    pc = 7'd0;
                    v_pack = 1536'd0;
                    for (l = 0; l < 64; l = l + 1) begin
                        co = (cfg_cout == 6'd16) ? (l & 15) : (l & 31);
                        g  = (cfg_cout == 6'd16) ? (l >> 4) : (l >> 5);
                        v_prev = {{2{v_rdata[24*l + 23]}}, v_rdata[24*l +: 24]};
                        cur = {{2{acc[l][23]}}, acc[l]} +
                              {{2{bias[co][23]}}, bias[co]} +
                              (cfg_sew && resvec[co * ((cfg_cout == 6'd16) ? 4 : 2) + g]
                               ? {{2{cfg_th[23]}}, cfg_th} : 26'sd0);
                        v_sum = v_prev + cur;
                        fire = (v_sum >= th_ext);
                        v_new = fire ? (v_sum - th_ext) : v_sum;
                        if (fire) begin
                            pc = pc + 7'd1;
                            obuf[co][group * ((cfg_cout == 6'd16) ? 4 : 2) + g] <= 1'b1;
                        end
                        v_pack[24*l +: 24] = v_new[23:0];
                    end
                    spk_count <= spk_count + {10'd0, pc};
                    v_we <= 1'b1;
                    v_waddr <= cfg_vbase + row * 8 + group;
                    v_wdata <= v_pack;
                    state <= S_NEXT_GRP;
                end

                S_NEXT_GRP: begin
                    if (group == 3'd7) begin
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
