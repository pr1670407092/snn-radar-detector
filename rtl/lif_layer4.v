`timescale 1ns/1ps
// One fully-connected SNN layer with 64 synapses/cycle, signed 4-bit weights.
//
// Weight memory layout: row-major, one destination neuron per row,
//   word address = row*W_WORDS + chunk,  chunk selects 64 consecutive inputs.
// Each 256-bit word holds 64 signed 4-bit weights, lane 0 in bits [3:0].
//
// Membrane state lives in a simple dual-port BRAM (one read + one write port)
// so a 704-neuron layer does not turn into a 17k-LUT read mux.  V is cleared
// by sweeping zeros through the write port at the start of an inference.
//
// Update rule (bit-exact with python/convert_snn.py):
//   v = v - (v >>> LEAK_SHIFT) + bias + sum(w_i * spike_i)
//   if v >= THRESH: spike, v = v - THRESH
module lif_layer4 #(
    parameter integer FANIN      = 704,
    parameter integer NEURONS    = 704,
    parameter integer W_WORDS    = FANIN / 64,
    parameter integer AW         = 13,
    parameter integer BW         = 10,
    parameter integer SPK_W      = FANIN,
    parameter integer THRESH     = 39,
    parameter integer LEAK_SHIFT = 0,
    parameter         BIAS_FILE  = "b1.mem"
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              clear,
    input  wire              start,
    input  wire [SPK_W-1:0]  input_spikes,
    output reg  [AW-1:0]     w_addr,
    input  wire [255:0]      w_data,
    output reg  [NEURONS-1:0] out_spikes,
    output reg               busy,
    output reg               done,
    output reg               clear_done
);
    localparam [2:0] S_IDLE = 3'd0, S_CLEAR = 3'd1, S_LOAD = 3'd2,
                     S_RUN  = 3'd3, S_DONE  = 3'd4, S_UPDATE = 3'd5;

    // ---------------- registration / state ----------------
    reg [2:0]        state;
    reg [BW-1:0]     row;
    reg [4:0]        chunk;
    reg [BW-1:0]     clear_cnt;
    reg [SPK_W-1:0]  spk_vec;

    // ---------------- membrane BRAM (simple dual port) ----------------
    reg signed [23:0] vmem [0:NEURONS-1];
    reg signed [23:0] v_q;
    reg signed [23:0] v_cur;

    reg               v_we;
    reg [BW-1:0]      v_waddr;
    reg signed [23:0] v_wdata;
    reg [BW-1:0]      v_raddr;

    always @(posedge clk) begin
        if (v_we)
            vmem[v_waddr] <= v_wdata;
        v_q <= vmem[v_raddr];
    end

    // ---------------- bias ROM (LUTRAM) ----------------
    (* ram_style = "distributed" *)
    reg signed [23:0] bmem [0:NEURONS-1];
    reg signed [23:0] bias_q;
    reg [BW-1:0]      bias_addr;

    initial begin
        $readmemh(BIAS_FILE, bmem);
    end

    always @(posedge clk) begin
        bias_q <= bmem[bias_addr];
    end

    // ---------------- 32-lane conditional-add tree ----------------
    wire [63:0] mask = spk_vec[chunk*64 +: 64];

    wire signed [3:0]  wlane [0:63];
    wire signed [15:0] p     [0:63];
    genvar gi;
    generate
        for (gi = 0; gi < 64; gi = gi + 1) begin : g_lane
            assign wlane[gi] = w_data[4*gi +: 4];
            assign p[gi] = mask[gi] ? {{12{wlane[gi][3]}}, wlane[gi]} : 16'sd0;
        end
    endgenerate

    wire signed [15:0] t1 [0:31];
    wire signed [15:0] t2 [0:15];
    wire signed [15:0] t3 [0:7];
    wire signed [15:0] t4 [0:3];
    wire signed [15:0] t5 [0:1];
    wire signed [15:0] t6;
    genvar gj;
    generate
        for (gj = 0; gj < 32; gj = gj + 1) assign t1[gj] = p[2*gj] + p[2*gj+1];
        for (gj = 0; gj < 16; gj = gj + 1) assign t2[gj] = t1[2*gj] + t1[2*gj+1];
        for (gj = 0; gj < 8;  gj = gj + 1) assign t3[gj] = t2[2*gj] + t2[2*gj+1];
        for (gj = 0; gj < 4;  gj = gj + 1) assign t4[gj] = t3[2*gj] + t3[2*gj+1];
        for (gj = 0; gj < 2;  gj = gj + 1) assign t5[gj] = t4[2*gj] + t4[2*gj+1];
    endgenerate
    assign t6 = t5[0] + t5[1];

    // ---------------- LIF update ----------------
    // acc_total is registered into `total` at the last chunk; the membrane
    // update then happens in S_UPDATE one cycle later.  This splits the
    // 64-lane adder tree and the 24-bit LIF update into two timing paths.
    reg signed [23:0] acc;
    reg signed [23:0] total;

    wire signed [23:0] acc_total = acc + {{8{t6[15]}}, t6};
    wire signed [23:0] update_current = total;
    wire signed [24:0] v_prev_w  = {{1{v_cur[23]}}, v_cur};
    wire signed [24:0] v_leak_w  = (LEAK_SHIFT == 0) ? v_prev_w :
                                   (v_prev_w - (v_prev_w >>> LEAK_SHIFT));
    wire signed [24:0] v_sum_w   = v_leak_w + {{1{update_current[23]}}, update_current};

    localparam signed [24:0] VMAX = 25'sd8388607;
    localparam signed [24:0] VMIN = -25'sd8388608;
    localparam signed [24:0] THRESH_W = THRESH;

    wire signed [24:0] v_sum_c = (v_sum_w > VMAX) ? VMAX :
                                 (v_sum_w < VMIN) ? VMIN : v_sum_w;
    wire               fire    = (v_sum_c >= THRESH_W);
    wire signed [24:0] v_new_w = fire ? (v_sum_c - THRESH_W) : v_sum_c;

    // ---------------- address generation ----------------
    // Weight/bias/V addresses are driven combinationally so the memories can
    // be prefetched one cycle ahead.  In S_LOAD we consume chunk 0 while
    // prefetching chunk 1; every S_RUN consumes chunk c and prefetches c+1.
    always @* begin
        w_addr    = {AW{1'b0}};
        bias_addr = {BW{1'b0}};
        v_raddr   = {BW{1'b0}};
        v_we      = 1'b0;
        v_waddr   = {BW{1'b0}};
        v_wdata   = 24'sd0;

        case (state)
            S_IDLE: begin
                if (start) begin
                    w_addr    = {AW{1'b0}};
                    bias_addr = {BW{1'b0}};
                    v_raddr   = {BW{1'b0}};
                end
            end
            S_LOAD: begin
                w_addr    = row * W_WORDS + 1;
                bias_addr = row;
                v_raddr   = (row == NEURONS-1) ? row : row + 1;
            end
            S_RUN: begin
                if (chunk == W_WORDS-1) begin
                    if (row == NEURONS-1) begin
                        w_addr    = row * W_WORDS + chunk;
                        bias_addr = row;
                        v_raddr   = row;
                    end else begin
                        w_addr    = (row + 1) * W_WORDS;
                        bias_addr = row + 1;
                        v_raddr   = row + 1;
                    end
                end else begin
                    w_addr    = row * W_WORDS + chunk + 1;
                    bias_addr = row;
                    v_raddr   = (row == NEURONS-1) ? row : row + 1;
                end
            end
            S_UPDATE: begin
                // Hold the just-prefetched next-row addresses for one extra
                // cycle, update the membrane, and write out_spikes[row].
                if (row == NEURONS-1) begin
                    w_addr    = row * W_WORDS + chunk;
                    bias_addr = row;
                    v_raddr   = row;
                end else begin
                    w_addr    = (row + 1) * W_WORDS;
                    bias_addr = row + 1;
                    v_raddr   = row + 1;
                end
                v_we    = 1'b1;
                v_waddr = row;
                v_wdata = v_new_w[23:0];
            end
            S_CLEAR: begin
                v_we    = 1'b1;
                v_waddr = clear_cnt;
                v_wdata = 24'sd0;
            end
            default: begin
            end
        endcase
    end

    // ---------------- control ----------------
    always @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            row        <= {BW{1'b0}};
            chunk      <= 5'd0;
            clear_cnt  <= {BW{1'b0}};
            out_spikes <= {NEURONS{1'b0}};
            busy       <= 1'b0;
            done       <= 1'b0;
            clear_done <= 1'b0;
            acc        <= 24'sd0;
            total      <= 24'sd0;
            v_cur      <= 24'sd0;
            spk_vec    <= {SPK_W{1'b0}};
        end else if (clear) begin
            state      <= S_CLEAR;
            row        <= {BW{1'b0}};
            chunk      <= 5'd0;
            clear_cnt  <= {BW{1'b0}};
            out_spikes <= {NEURONS{1'b0}};
            busy       <= 1'b0;
            done       <= 1'b0;
            clear_done <= 1'b0;
            acc        <= 24'sd0;
            total      <= 24'sd0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        spk_vec    <= input_spikes;
                        row        <= {BW{1'b0}};
                        chunk      <= 5'd0;
                        busy       <= 1'b1;
                        done       <= 1'b0;
                        clear_done <= 1'b0;
                        state      <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    acc     <= bias_q + {{8{t6[15]}}, t6};
                    v_cur   <= v_q;
                    chunk   <= 5'd1;
                    state   <= S_RUN;
                end
                S_RUN: begin
                    if (chunk == W_WORDS-1) begin
                        total <= acc_total;
                        state <= S_UPDATE;
                    end else begin
                        acc   <= acc + {{8{t6[15]}}, t6};
                        chunk <= chunk + 1;
                    end
                end
                S_UPDATE: begin
                    out_spikes[row] <= fire;
                    if (row == NEURONS-1) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        row   <= row + 1;
                        chunk <= 5'd0;
                        state <= S_LOAD;
                    end
                end
                S_DONE: begin
                    done  <= 1'b0;
                    state <= S_IDLE;
                end
                S_CLEAR: begin
                    if (clear_cnt == NEURONS-1) begin
                        clear_done <= 1'b1;
                        state      <= S_IDLE;
                    end else begin
                        clear_cnt <= clear_cnt + 1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
