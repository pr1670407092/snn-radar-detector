`timescale 1ns/1ps
// Board self-test wrapper for 正点原子 达芬奇Pro (Artix-7 XC7A200T-2FBG484I).
// Pure PL design: no PS, no DDR, no MIG.  JTAG 下载后按复位键即可自检。
//
// - 50 MHz board clock -> MMCM -> 100 MHz core clock
// - runs the 8 test vectors stored in feat_all.mem, compares decisions with
//   exp_dec.mem, and reports pass_count on LEDs
// - no PS/DDR required; press reset to run again
//
// LED: [0] core busy, [1] all 8 cases done, [2] last decision,
//      [3] pass_count == 8
module board_top (
    input  wire       clk50,
    input  wire       rst_n,
    output reg  [3:0] led
);
    // ---------------- 50 MHz -> 100 MHz ----------------
    wire clk100_raw, clk100, clkfb, locked;
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKIN1_PERIOD(20.000),
        .CLKFBOUT_MULT_F(20.000),
        .DIVCLK_DIVIDE(1),
        .CLKOUT0_DIVIDE_F(10.000),
        .CLKOUT0_DUTY_CYCLE(0.500),
        .CLKOUT0_PHASE(0.000),
        .STARTUP_WAIT("FALSE")
    ) u_mmcm (
        .CLKOUT0(clk100_raw),
        .CLKFBOUT(clkfb),
        .CLKFBIN(clkfb),
        .CLKIN1(clk50),
        .PWRDWN(1'b0),
        .RST(1'b0),
        .LOCKED(locked)
    );
    BUFG u_bufg (.I(clk100_raw), .O(clk100));

    // reset synchronizer (rst_n is active-low)
    reg [2:0] rst_sync;
    always @(posedge clk100)
        rst_sync <= {rst_sync[1:0], rst_n};
    wire core_rst = ~rst_sync[2] | ~locked;

    // ---------------- test-vector ROMs ----------------
    // NOTE: exp_rom 必须是多位宽 memory。原工程写的是 `reg exp_rom [0:7]`（1 bit），
    // Vivado 综合会报 "Synth 8-2898 ignoring malformed $readmem task: invalid memory name"
    // 并**静默跳过初始化**，硬件上 exp_rom 全 0（期望 1,1,1,1,0,0,0,0），
    // 导致 pass_count 只有 4、LED3 不亮。改成 [7:0] 后综合器才会真正预置 ROM。
    reg [7:0] feat_rom [0:8191];
    reg [7:0] exp_rom  [0:7];
    initial begin
        $readmemh("feat_all.mem", feat_rom);
        $readmemh("exp_dec.mem",  exp_rom);
    end

    // ---------------- core ----------------
    reg         feat_we;
    reg  [9:0]  feat_addr;
    reg  [7:0]  feat_data;
    reg         start;
    wire        busy, done, decision;
    wire signed [47:0] score0, score1;
    wire [2:0]  dbg_layer;
    wire [1:0]  dbg_t;
    wire [3:0]  dbg_state;
    wire [17:0] dbg_count0, dbg_count1, dbg_count2;
    wire [17:0] dbg_count3, dbg_count4, dbg_count5;

    snn2d_top u_core (
        .clk(clk100), .rst(core_rst), .start(start),
        .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
        .busy(busy), .done(done), .decision(decision),
        .score0(score0), .score1(score1),
        .dbg_layer(dbg_layer), .dbg_t(dbg_t), .dbg_state(dbg_state),
        .dbg_count0(dbg_count0), .dbg_count1(dbg_count1), .dbg_count2(dbg_count2),
        .dbg_count3(dbg_count3), .dbg_count4(dbg_count4), .dbg_count5(dbg_count5)
    );

    // ---------------- self-test FSM ----------------
    localparam S_IDLE  = 3'd0,
               S_LOAD  = 3'd1,
               S_START = 3'd2,
               S_WAIT  = 3'd3,
               S_REC   = 3'd4,
               S_DONE  = 3'd5;
    reg [2:0]  state;
    reg [2:0]  case_idx;
    reg [9:0]  load_cnt;
    reg [3:0]  pass_count;
    reg        done_all;
    reg        last_dec;

    wire [12:0] rom_addr = {case_idx, load_cnt};

    always @(posedge clk100) begin
        if (core_rst) begin
            state <= S_IDLE;
            case_idx <= 3'd0;
            load_cnt <= 10'd0;
            pass_count <= 4'd0;
            done_all <= 1'b0;
            last_dec <= 1'b0;
            feat_we <= 1'b0;
            feat_addr <= 10'd0;
            feat_data <= 8'd0;
            start <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    feat_we <= 1'b0;
                    start <= 1'b0;
                    if (locked) begin
                        case_idx <= 3'd0;
                        load_cnt <= 10'd0;
                        pass_count <= 4'd0;
                        done_all <= 1'b0;
                        state <= S_LOAD;
                    end
                end
                S_LOAD: begin
                    feat_we <= 1'b1;
                    feat_addr <= load_cnt;
                    feat_data <= feat_rom[rom_addr];
                    if (load_cnt == 10'd1023) begin
                        feat_we <= 1'b0;
                        state <= S_START;
                    end else begin
                        load_cnt <= load_cnt + 10'd1;
                    end
                end
                S_START: begin
                    start <= 1'b1;
                    state <= S_WAIT;
                end
                S_WAIT: begin
                    start <= 1'b0;
                    if (done)
                        state <= S_REC;
                end
                S_REC: begin
                    last_dec <= decision;
                    if (decision == exp_rom[case_idx][0])
                        pass_count <= pass_count + 4'd1;
                    if (case_idx == 3'd7) begin
                        done_all <= 1'b1;
                        state <= S_DONE;
                    end else begin
                        case_idx <= case_idx + 3'd1;
                        load_cnt <= 10'd0;
                        state <= S_LOAD;
                    end
                end
                S_DONE: begin
                    start <= 1'b0;
                    feat_we <= 1'b0;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // 0 = 高电平点亮（正点原子达芬奇Pro 4 个用户 LED 默认如此）
    // 若上板发现 LED 全反，改成 1 重新综合。
    localparam LED_ACTIVE_LOW = 1'b0;

    always @* begin
        led[0] = busy             ^ LED_ACTIVE_LOW;
        led[1] = done_all         ^ LED_ACTIVE_LOW;
        led[2] = last_dec         ^ LED_ACTIVE_LOW;
        led[3] = (pass_count == 4'd8) ^ LED_ACTIVE_LOW;
    end
endmodule

