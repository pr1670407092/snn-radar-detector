`timescale 1ns/1ps
// Self-checking testbench for the 2D time-frequency RSN accelerator.
// Compares decision, FC scores and per-layer spike counts against the
// bit-exact integer Python reference.
module tb_snn2d_top;
    localparam integer N_CASES = 8;

    reg clk, rst, start;
    reg feat_we;
    reg [9:0] feat_addr;
    reg [7:0] feat_data;
    wire busy, done, decision;
    wire signed [47:0] score0, score1;
    wire [2:0] dbg_layer;
    wire [1:0] dbg_t;
    wire [3:0] dbg_state;
    wire [17:0] dbg_count0, dbg_count1, dbg_count2, dbg_count3, dbg_count4, dbg_count5;

    reg [7:0] feat_img [0:1023];
    reg [63:0] exp_score0 [0:N_CASES-1];
    reg [63:0] exp_score1 [0:N_CASES-1];
    reg [16:0] exp_c0 [0:N_CASES-1];
    reg [16:0] exp_c1 [0:N_CASES-1];
    reg [16:0] exp_c2 [0:N_CASES-1];
    reg [16:0] exp_c3 [0:N_CASES-1];
    reg [16:0] exp_c4 [0:N_CASES-1];
    reg [16:0] exp_c5 [0:N_CASES-1];
    reg [7:0]  exp_dec [0:N_CASES-1];

    integer ci, k, errors, timeout;
    integer dco, drow, dfd, dfd2, dfd3, dfd4, dfd5, dfd6, vg;
    integer enc_count;
    reg [63:0] got0, got1;

    function integer pop32;
        input [31:0] v;
        integer i;
        begin
            pop32 = 0;
            for (i = 0; i < 32; i = i + 1)
                pop32 = pop32 + v[i];
        end
    endfunction

    always @(posedge clk)
        if (dut.u_enc.spk_we)
            enc_count = enc_count + pop32(dut.u_enc.spk_data);


    snn2d_top dut (
        .clk(clk), .rst(rst), .start(start),
        .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
        .busy(busy), .done(done), .decision(decision),
        .score0(score0), .score1(score1),
        .dbg_layer(dbg_layer), .dbg_t(dbg_t), .dbg_state(dbg_state),
        .dbg_count0(dbg_count0), .dbg_count1(dbg_count1), .dbg_count2(dbg_count2),
        .dbg_count3(dbg_count3), .dbg_count4(dbg_count4), .dbg_count5(dbg_count5)
    );

    always #5 clk = ~clk;

    task load_feat(input integer c);
        begin
            case (c)
                0: $readmemh("feat_0.mem", feat_img);
                1: $readmemh("feat_1.mem", feat_img);
                2: $readmemh("feat_2.mem", feat_img);
                3: $readmemh("feat_3.mem", feat_img);
                4: $readmemh("feat_4.mem", feat_img);
                5: $readmemh("feat_5.mem", feat_img);
                6: $readmemh("feat_6.mem", feat_img);
                7: $readmemh("feat_7.mem", feat_img);
            endcase
            for (k = 0; k < 1024; k = k + 1) begin
                @(posedge clk);
                feat_we <= 1'b1;
                feat_addr <= k[9:0];
                feat_data <= feat_img[k];
            end
            @(posedge clk);
            feat_we <= 1'b0;
        end
    endtask

    initial begin
        $readmemh("expected_score0.mem", exp_score0);
        $readmemh("expected_score1.mem", exp_score1);
        $readmemh("expected_counts_0.mem", exp_c0);
        $readmemh("expected_counts_1.mem", exp_c1);
        $readmemh("expected_counts_2.mem", exp_c2);
        $readmemh("expected_counts_3.mem", exp_c3);
        $readmemh("expected_counts_4.mem", exp_c4);
        $readmemh("expected_counts_5.mem", exp_c5);
        $readmemh("expected_dec.mem", exp_dec);

        clk = 1'b0;
        rst = 1'b1;
        start = 1'b0;
        feat_we = 1'b0;
        feat_addr = 10'd0;
        feat_data = 8'd0;
        errors = 0;

        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        for (ci = 0; ci < N_CASES; ci = ci + 1) begin
            load_feat(ci);
            enc_count = 0;

            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;

            timeout = 0;
            while (!done && timeout < 2000000) begin
                @(posedge clk);
                #1;
                timeout = timeout + 1;
            end
            if (!done) begin
                $display("CASE %0d FAIL: timeout (state=%0d layer=%0d t=%0d)",
                         ci, dbg_state, dbg_layer, dbg_t);
                $display("  enc: state=%0d k=%0d bitpos=%0d done=%0d clear_done=%0d busy=%0d",
                         dut.u_enc.state, dut.u_enc.k, dut.u_enc.bitpos,
                         dut.u_enc.done, dut.u_enc.clear_done, dut.u_enc.busy);
                $display("  top: enc_start=%0d enc_clear=%0d tstate=%0d",
                         dut.enc_start, dut.enc_clear, dut.tstate);
                errors = errors + 1;
            end else begin
                got0 = {{16{score0[47]}}, score0};
                got1 = {{16{score1[47]}}, score1};
                if (decision !== exp_dec[ci][0]) begin
                    $display("CASE %0d FAIL: dec=%0d exp=%0d score0=%0d exp=%0d score1=%0d exp=%0d",
                             ci, decision, exp_dec[ci],
                             $signed(got0), $signed(exp_score0[ci]),
                             $signed(got1), $signed(exp_score1[ci]));
                    $display("  enc=%0d counts: %0d/%0d %0d/%0d %0d/%0d %0d/%0d %0d/%0d %0d/%0d",
                             enc_count,
                             dbg_count0, exp_c0[ci], dbg_count1, exp_c1[ci],
                             dbg_count2, exp_c2[ci], dbg_count3, exp_c3[ci],
                             dbg_count4, exp_c4[ci], dbg_count5, exp_c5[ci]);
                    errors = errors + 1;
                end else begin
                    $display("CASE %0d PASS: dec=%0d score0=%0d score1=%0d enc=%0d counts=%0d/%0d/%0d/%0d/%0d/%0d",
                             ci, decision, $signed(got0), $signed(got1), enc_count,
                             dbg_count0, dbg_count1, dbg_count2,
                             dbg_count3, dbg_count4, dbg_count5);
                end
            end
            repeat (3) @(posedge clk);
        end

        if (errors == 0)
            $display("ALL %0d CASES PASS", N_CASES);
        else
            $display("%0d ERRORS", errors);
        $finish;
    end

    initial begin
        #200000000; // 200 ms simulation guard
        $display("GLOBAL TIMEOUT");
        $finish;
    end
endmodule
