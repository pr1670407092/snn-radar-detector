// Self-checking testbench for the efficient 8-bit 512-256-128-2 SNN.
// and compares the output spike counts / decision against the bit-exact
// Python reference (weights/hw_expected.mem).
`timescale 1ns / 1ps

module tb_snn_top;
    localparam integer N_CASES = 8;
    localparam integer N_FEAT  = 512;
    localparam integer T       = 16;

    reg         clk;
    reg         rst;
    reg         feat_we;
    reg  [8:0]  feat_addr;
    reg  [7:0]  feat_data;
    reg         start;
    wire        busy;
    wire        done;
    wire        decision;
    wire [15:0] count0;
    wire [15:0] count1;

    reg [4095:0] feat_words [0:N_CASES-1];
    reg [63:0]   exp_words  [0:N_CASES-1];

    integer c, b, errors;
    integer cyc;
    reg        exp_dec;
    reg [15:0] exp_c1, exp_c0;

    // Parameter defaults in snn_top.v were generated to match weights/meta.json.
    // T is overridden here only to make the intent explicit.
    snn_top_eff #(.T(T)) dut (
        .clk(clk), .rst(rst),
        .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
        .start(start),
        .busy(busy), .done(done), .decision(decision),
        .count0(count0), .count1(count1)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk) if (dut.busy) cyc <= cyc + 1;


    initial begin
        $display("=== SNN hardware simulation: %0d cases, T=%0d ===", N_CASES, T);
        $readmemh("hw_features.mem", feat_words);
        $readmemh("hw_expected.mem", exp_words);

        rst      = 1'b1;
        feat_we  = 1'b0;
        feat_addr = 9'd0;
        feat_data = 8'd0;
        start    = 1'b0;
        errors   = 0;

        repeat (10) @(posedge clk);
        rst = 1'b0;
        repeat (5) @(posedge clk);

        for (c = 0; c < N_CASES; c = c + 1) begin
            // load 256 feature bytes
            for (b = 0; b < N_FEAT; b = b + 1) begin
                @(negedge clk);
                feat_we   = 1'b1;
                feat_addr = b[8:0];
                feat_data = feat_words[c][8*b +: 8];
            end
            @(negedge clk);
            feat_we = 1'b0;

            @(negedge clk);
            cyc = 0;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            while (!done) @(negedge clk);

            exp_dec = exp_words[c][0];
            exp_c1  = exp_words[c][31:16];
            exp_c0  = exp_words[c][47:32];

            if ((decision === exp_dec) && (count1 === exp_c1) && (count0 === exp_c0)) begin
                $display("CASE %0d PASS: decision=%0d count0=%0d count1=%0d cycles=%0d",
                         c, decision, count0, count1, cyc);
            end else begin
                $display("CASE %0d FAIL: got decision=%0d count0=%0d count1=%0d | expected decision=%0d count0=%0d count1=%0d",
                         c, decision, count0, count1, exp_dec, exp_c0, exp_c1);
                errors = errors + 1;
            end

            repeat (5) @(negedge clk);
        end

        if (errors == 0)
            $display("=== RESULT: PASS (%0d/%0d cases) ===", N_CASES, N_CASES);
        else
            $display("=== RESULT: FAIL (%0d/%0d cases failed) ===", errors, N_CASES);
        $finish;
    end

    initial begin
        #100_000_000;
        $display("=== RESULT: TIMEOUT ===");
        $finish;
    end
endmodule
