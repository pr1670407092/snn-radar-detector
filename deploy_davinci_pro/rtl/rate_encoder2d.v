`timescale 1ns/1ps
// Sequential 1024-channel sigma-delta rate encoder (one pixel per cycle).
// Keeps the phase accumulator across timesteps; writes one 32-bit spike row
// word per 32 pixels into the global spike memory (base 0).
module rate_encoder2d #(
    parameter integer N        = 1024,
    parameter integer AW       = 10,
    parameter integer INIT_MUL = 73
) (
    input  wire           clk,
    input  wire           rst,
    input  wire           clear,
    input  wire           start,
    input  wire           feat_we,
    input  wire [AW-1:0]  feat_addr,
    input  wire [7:0]     feat_data,
    output reg            busy,
    output reg            done,
    output reg            clear_done,
    output reg            spk_we,
    output reg  [4:0]     spk_row,
    output reg  [31:0]    spk_data
);
    localparam S_IDLE  = 2'd0,
               S_CLEAR = 2'd1,
               S_RUN   = 2'd2,
               S_DONE  = 2'd3;

    reg [7:0] feat [0:N-1];
    reg [8:0] acc  [0:N-1];
    reg [1:0] state;
    reg [AW-1:0] k;
    reg [4:0] bitpos;
    reg [31:0] rowbuf;
    reg [8:0] s;
    reg fire;

    always @(posedge clk) begin
        if (feat_we)
            feat[feat_addr] <= feat_data;
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            clear_done <= 1'b0;
            spk_we <= 1'b0;
            spk_row <= 5'd0;
            spk_data <= 32'd0;
            k <= {AW{1'b0}};
            bitpos <= 5'd0;
            rowbuf <= 32'd0;
        end else begin
            spk_we <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    done <= 1'b0;
                    if (clear) begin
                        state <= S_CLEAR;
                        k <= {AW{1'b0}};
                        clear_done <= 1'b0;
                        busy <= 1'b1;
                    end else if (start) begin
                        state <= S_RUN;
                        k <= {AW{1'b0}};
                        bitpos <= 5'd0;
                        rowbuf <= 32'd0;
                        busy <= 1'b1;
                        done <= 1'b0;
                    end
                end

                S_CLEAR: begin
                    acc[k] <= (k * INIT_MUL) & 255;
                    if (k == N-1) begin
                        clear_done <= 1'b1;
                        busy <= 1'b0;
                        state <= S_IDLE;
                    end else begin
                        k <= k + {{(AW-1){1'b0}}, 1'b1};
                    end
                end

                S_RUN: begin
                    s = acc[k] + {1'b0, feat[k]};
                    fire = (s >= 9'd256);
                    acc[k] <= fire ? (s - 9'd256) : s;
                    rowbuf[bitpos] <= fire;
                    if (bitpos == 5'd31) begin
                        spk_we <= 1'b1;
                        spk_row <= k[AW-1:5];
                        spk_data <= {fire, rowbuf[30:0]};
                        bitpos <= 5'd0;
                        rowbuf <= 32'd0;
                        if (k == N-1) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= S_DONE;
                        end else begin
                            k <= k + {{(AW-1){1'b0}}, 1'b1};
                        end
                    end else begin
                        bitpos <= bitpos + 5'd1;
                        k <= k + {{(AW-1){1'b0}}, 1'b1};
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                    if (start && !clear) begin
                        k <= {AW{1'b0}};
                        bitpos <= 5'd0;
                        rowbuf <= 32'd0;
                        busy <= 1'b1;
                        done <= 1'b0;
                        state <= S_RUN;
                    end
                end
            endcase
        end
    end
endmodule
