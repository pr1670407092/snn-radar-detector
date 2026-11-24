`timescale 1ns/1ps
// Self-checking testbench for the board-level wrapper (deploy/rtl/board_top.v).
//
// The core testbench (tb_snn2d_top.v) only proves the accelerator computes the
// right decisions.  It says nothing about the board shell — ROM addressing, the
// self-test FSM, or the LED mapping.  Those are exactly where board-level bugs
// hide, so they need their own testbench.
//
// This TB drives the real 50 MHz board clock, lets the MMCM lock, runs all 8
// cases and then checks:
//   * every decision against the expected-decision ROM
//   * pass_count == 8
//   * the final LED pattern matches what the README promises
//
// Requires the Xilinx unisim models (MMCME2_ADV, MMCME2_BASE, BUFG) and glbl;
// see sim/run_board_sim.sh for the exact compile/elaborate commands.
module tb_board_top;
    reg clk50 = 1'b0;
    reg rst_n = 1'b0;
    wire [3:0] led;

    integer errors = 0;
    integer t0, elapsed, guard, i;

    board_top dut (.clk50(clk50), .rst_n(rst_n), .led(led));

    // 50 MHz board oscillator (20 ns period)
    always #10 clk50 = ~clk50;

    // Sample the decision of each case as done rises
    reg past_done = 1'b0;
    integer n_done = 0;
    reg [7:0] dec_seen [0:7];
    always @(posedge dut.clk100) begin
        past_done <= dut.done;
        if (dut.done && !past_done && n_done < 8) begin
            dec_seen[n_done] = {7'b0, dut.decision};
            n_done = n_done + 1;
        end
    end

    initial begin
        $display("=== tb_board_top: releasing reset ===");
        repeat (20) @(posedge clk50);
        rst_n = 1'b1;
        t0 = $time;

        // One case takes ~5.82 ms, so 8 cases plus feature loading is ~47 ms.
        // Budget from that: 47 ms / 20 ns ~= 2.4M clocks.  Poll up to 120 ms.
        guard = 0;
        while (led[1] !== 1'b1 && guard < 6000000) begin
            @(posedge clk50);
            guard = guard + 1;
        end
        elapsed = $time - t0;

        if (led[1] !== 1'b1) begin
            $display("*** TIMEOUT: done_all never asserted (%0d ms) ***", elapsed/1000000);
            errors = errors + 1;
        end else begin
            $display("done_all asserted after %0d ns (%0d ms)", elapsed, elapsed/1000000);
        end

        repeat (8) @(posedge dut.clk100);

        $display("cases done     = %0d", n_done);
        $display("decisions seen = %0d %0d %0d %0d %0d %0d %0d %0d",
                 dec_seen[0], dec_seen[1], dec_seen[2], dec_seen[3],
                 dec_seen[4], dec_seen[5], dec_seen[6], dec_seen[7]);
        $display("LED = %b  (led0=busy led1=done_all led2=last_dec led3=pass==8)", led);
        $display("pass_count = %0d   last_dec = %0d", dut.pass_count, dut.last_dec);

        for (i = 0; i < 8; i = i + 1) begin
            if (dec_seen[i] !== {7'b0, dut.exp_rom[i][0]}) begin
                $display("*** FAIL: case %0d decision = %0d, expected %0d ***",
                         i, dec_seen[i], dut.exp_rom[i][0]);
                errors = errors + 1;
            end
        end

        if (dut.pass_count !== 4'd8) begin
            $display("*** FAIL: pass_count = %0d, expected 8 ***", dut.pass_count);
            errors = errors + 1;
        end
        if (led[3] !== 1'b1) begin $display("*** FAIL: led[3] should be 1 ***"); errors = errors + 1; end
        if (led[1] !== 1'b1) begin $display("*** FAIL: led[1] should be 1 ***"); errors = errors + 1; end
        if (led[2] !== 1'b0) begin $display("*** FAIL: led[2] should be 0 (case 7 is negative) ***"); errors = errors + 1; end
        if (led[0] !== 1'b0) begin $display("*** FAIL: led[0] should be 0 (core idle) ***"); errors = errors + 1; end

        if (errors == 0)
            $display("=== BOARD TOP SELF-TEST PASS: 8/8, LED pattern exactly as documented ===");
        else
            $display("=== BOARD TOP SELF-TEST: %0d ERROR(S) ===", errors);
        $finish;
    end
endmodule
