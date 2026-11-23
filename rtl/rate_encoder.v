`timescale 1ns/1ps
// First-order sigma-delta rate encoder.
//
// At every tick:
//     s    = acc + feat
//     fire = (s >= 256)
//     acc  = fire ? s-256 : s
// This keeps the long-run spike rate exactly feat/256 and spreads spikes in
// time better than a simple phase comparator for short T.
//
// acc is pre-loaded with (k*73)&255 at the start of every inference so that
// different frequency bins do not all fire on the same timestep.  python/
// convert_snn.py:encode() reproduces this bit-exactly.
module rate_encoder #(
    parameter integer N        = 512,
    parameter integer FAW      = 9,
    parameter integer INIT_MUL = 73
) (
    input  wire           clk,
    input  wire           rst,
    input  wire           clear,
    input  wire           tick,
    input  wire           feat_we,
    input  wire [FAW-1:0] feat_addr,
    input  wire [7:0]     feat_data,
    output reg  [N-1:0]   spikes
);
    reg [7:0] feat [0:N-1];
    reg [8:0] acc  [0:N-1];
    integer k;

    always @(posedge clk) begin
        if (rst) begin
            spikes <= {N{1'b0}};
            for (k = 0; k < N; k = k + 1)
                acc[k] <= (k * INIT_MUL) & 255;
        end else begin
            if (feat_we)
                feat[feat_addr] <= feat_data;

            if (clear) begin
                spikes <= {N{1'b0}};
                for (k = 0; k < N; k = k + 1)
                    acc[k] <= (k * INIT_MUL) & 255;
            end else if (tick) begin
                for (k = 0; k < N; k = k + 1) begin
                    if ((acc[k] + {1'b0, feat[k]}) >= 9'd256) begin
                        spikes[k] <= 1'b1;
                        acc[k]    <= acc[k] + {1'b0, feat[k]} - 9'd256;
                    end else begin
                        spikes[k] <= 1'b0;
                        acc[k]    <= acc[k] + {1'b0, feat[k]};
                    end
                end
            end
        end
    end
endmodule
