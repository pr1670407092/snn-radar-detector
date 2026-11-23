`timescale 1ns/1ps
// Wide int8 weight memory: 32 weights per 256-bit word.
// Read latency = 1 clock.  Initialised from a $readmemh file.
module weight_mem #(
    parameter integer DEPTH = 4096,
    parameter integer AW    = 12,
    parameter         FILE  = "w1.mem"
) (
    input  wire         clk,
    input  wire [AW-1:0] addr,
    output reg  [255:0] dout
);
    (* ram_style = "block" *)
    reg [255:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(FILE, mem);
    end

    always @(posedge clk) begin
        dout <= mem[addr];
    end
endmodule

// Small weight memory for the 2-neuron output layer: LUTRAM is cheaper than
// wasting 8 BRAMs on 44 words.
module weight_mem_lut #(
    parameter integer DEPTH = 44,
    parameter integer AW    = 6,
    parameter         FILE  = "w3.mem"
) (
    input  wire         clk,
    input  wire [AW-1:0] addr,
    output reg  [255:0] dout
);
    (* ram_style = "distributed" *)
    reg [255:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(FILE, mem);
    end

    always @(posedge clk) begin
        dout <= mem[addr];
    end
endmodule
