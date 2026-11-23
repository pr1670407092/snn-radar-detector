`timescale 1ns/1ps
// 3-layer fully-connected SNN accelerator for the ALINETK "领航者" ZYNQ-7020.
//
//   input (256 freq bins, sigma-delta rate coded)
//     -> 512 LIF
//     -> 704 LIF
//     -> 2 LIF (no-target / target)
//
// 32 synapses are processed per clock.  int8 weights, no multipliers: a weight
// is conditionally added when its pre-synaptic spike is 1.  All neuron state
// is integer and bit-exact with python/convert_snn.py.
//
// Weight files are loaded by $readmemh at time 0; the Python exporter writes
// them in the exact RTL layout.  Parameter defaults below are generated to
// match weights/meta.json (see rtl/snn_params.vh).
module snn_top #(
    parameter integer T          = 16,
    parameter integer TH1        = 316,
    parameter integer TH2        = 289,
    parameter integer TH3        = 3571,
    parameter integer LEAK_SHIFT = 0
) (
    input  wire        clk,
    input  wire        rst,
    // feature load port (256 bytes, one byte per clock)
    input  wire        feat_we,
    input  wire [8:0]  feat_addr,
    input  wire [7:0]  feat_data,
    // control / result
    input  wire        start,
    output reg         busy,
    output reg         done,
    output reg         decision,
    output reg  [15:0] count0,
    output reg  [15:0] count1
);
    localparam integer N_IN  = 512;
    localparam integer N_H1  = 512;
    localparam integer N_H2  = 512;
    localparam integer N_OUT = 2;

    localparam integer W1_WORDS = N_H1 * (N_IN / 32);   // 8192
    localparam integer W2_WORDS = N_H2 * (N_H1 / 32);   // 8192
    localparam integer W3_WORDS = N_OUT * (N_H2 / 32);  // 32

    localparam integer AW1 = 13, AW2 = 13, AW3 = 5;
    localparam integer BW1 = 9,  BW2 = 9,  BW3 = 1;

    // ---------------- top-level sequencer state ----------------
    localparam [3:0] S_IDLE     = 4'd0,
                     S_CLEAR    = 4'd1,
                     S_WAIT     = 4'd2,
                     S_TICK     = 4'd3,
                     S_L1START  = 4'd4,
                     S_L1WAIT   = 4'd5,
                     S_L2START  = 4'd6,
                     S_L2WAIT   = 4'd7,
                     S_L3START  = 4'd8,
                     S_L3WAIT   = 4'd9,
                     S_DONE     = 4'd10,
                     S_TICK2    = 4'd11;

    reg [3:0]  state;
    reg [7:0]  t;

    // ---------------- encoder ----------------
    // Registered + replicated control: the 512x9 encoder accumulators present
    // a large clock-enable fanout, so drive them from replicated registers.
    wire [N_IN-1:0] enc_spikes;
    (* max_fanout = 16 *) reg enc_clear_r;
    (* max_fanout = 16 *) reg enc_tick_r;

    always @(posedge clk) begin
        if (rst) begin
            enc_clear_r <= 1'b0;
            enc_tick_r  <= 1'b0;
        end else begin
            enc_clear_r <= (state == S_CLEAR);
            enc_tick_r  <= (state == S_TICK);
        end
    end

    rate_encoder #(.N(N_IN), .FAW(9)) u_enc (
        .clk(clk), .rst(rst), .clear(enc_clear_r), .tick(enc_tick_r),
        .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
        .spikes(enc_spikes)
    );

    // ---------------- weight memories ----------------
    wire [255:0] w1_data, w2_data, w3_data;
    wire [AW1-1:0] w1_addr;
    wire [AW2-1:0] w2_addr;
    wire [AW3-1:0] w3_addr;

    weight_mem #(.DEPTH(W1_WORDS), .AW(AW1), .FILE("w1.mem")) u_wmem1 (
        .clk(clk), .addr(w1_addr), .dout(w1_data)
    );
    weight_mem #(.DEPTH(W2_WORDS), .AW(AW2), .FILE("w2.mem")) u_wmem2 (
        .clk(clk), .addr(w2_addr), .dout(w2_data)
    );
    weight_mem_lut #(.DEPTH(W3_WORDS), .AW(AW3), .FILE("w3.mem")) u_wmem3 (
        .clk(clk), .addr(w3_addr), .dout(w3_data)
    );

    // ---------------- layers ----------------
    wire [N_H1-1:0]  l1_spikes;
    wire [N_H2-1:0]  l2_spikes;
    wire [N_OUT-1:0] l3_spikes;
    wire l1_done, l2_done, l3_done;
    wire l1_clear_done, l2_clear_done, l3_clear_done;

    (* max_fanout = 16 *) wire l1_start = (state == S_L1START);
    (* max_fanout = 16 *) wire l2_start = (state == S_L2START);
    (* max_fanout = 16 *) wire l3_start = (state == S_L3START);
    (* max_fanout = 16 *) wire l1_clear = (state == S_CLEAR);
    (* max_fanout = 16 *) wire l2_clear = (state == S_CLEAR);
    (* max_fanout = 16 *) wire l3_clear = (state == S_CLEAR);

    lif_layer #(
        .FANIN(N_IN), .NEURONS(N_H1), .W_WORDS(N_IN/32),
        .AW(AW1), .BW(BW1), .SPK_W(N_IN),
        .THRESH(TH1), .LEAK_SHIFT(LEAK_SHIFT), .BIAS_FILE("b1.mem")
    ) u_l1 (
        .clk(clk), .rst(rst), .clear(l1_clear), .start(l1_start),
        .input_spikes(enc_spikes),
        .w_addr(w1_addr), .w_data(w1_data),
        .out_spikes(l1_spikes), .busy(), .done(l1_done),
        .clear_done(l1_clear_done)
    );

    lif_layer #(
        .FANIN(N_H1), .NEURONS(N_H2), .W_WORDS(N_H1/32),
        .AW(AW2), .BW(BW2), .SPK_W(N_H1),
        .THRESH(TH2), .LEAK_SHIFT(LEAK_SHIFT), .BIAS_FILE("b2.mem")
    ) u_l2 (
        .clk(clk), .rst(rst), .clear(l2_clear), .start(l2_start),
        .input_spikes(l1_spikes),
        .w_addr(w2_addr), .w_data(w2_data),
        .out_spikes(l2_spikes), .busy(), .done(l2_done),
        .clear_done(l2_clear_done)
    );

    lif_layer #(
        .FANIN(N_H2), .NEURONS(N_OUT), .W_WORDS(N_H2/32),
        .AW(AW3), .BW(BW3), .SPK_W(N_H2),
        .THRESH(TH3), .LEAK_SHIFT(LEAK_SHIFT), .BIAS_FILE("b3.mem")
    ) u_l3 (
        .clk(clk), .rst(rst), .clear(l3_clear), .start(l3_start),
        .input_spikes(l2_spikes),
        .w_addr(w3_addr), .w_data(w3_data),
        .out_spikes(l3_spikes), .busy(), .done(l3_done),
        .clear_done(l3_clear_done)
    );

    // ---------------- top-level sequencer ----------------
    wire [15:0] next_cnt0 = count0 + {15'b0, l3_spikes[0]};
    wire [15:0] next_cnt1 = count1 + {15'b0, l3_spikes[1]};

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            t        <= 8'd0;
            count0   <= 16'd0;
            count1   <= 16'd0;
            busy     <= 1'b0;
            done     <= 1'b0;
            decision <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy     <= 1'b1;
                        done     <= 1'b0;
                        count0   <= 16'd0;
                        count1   <= 16'd0;
                        t        <= 8'd0;
                        state    <= S_CLEAR;
                    end
                end
                S_CLEAR: begin
                    state <= S_WAIT;
                end
                S_WAIT: begin
                    if (l1_clear_done && l2_clear_done && l3_clear_done)
                        state <= S_TICK;
                end
                S_TICK: begin
                    state <= S_TICK2;
                end
                S_TICK2: begin
                    state <= S_L1START;
                end
                S_L1START: begin
                    state <= S_L1WAIT;
                end
                S_L1WAIT: begin
                    if (l1_done) state <= S_L2START;
                end
                S_L2START: begin
                    state <= S_L2WAIT;
                end
                S_L2WAIT: begin
                    if (l2_done) state <= S_L3START;
                end
                S_L3START: begin
                    state <= S_L3WAIT;
                end
                S_L3WAIT: begin
                    if (l3_done) begin
                        count0 <= next_cnt0;
                        count1 <= next_cnt1;
                        if (t == T-1) begin
                            decision <= (next_cnt1 > next_cnt0);
                            busy     <= 1'b0;
                            done     <= 1'b1;
                            state    <= S_DONE;
                        end else begin
                            t     <= t + 8'd1;
                            state <= S_TICK;
                        end
                    end
                end
                S_DONE: begin
                    done  <= 1'b0;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
