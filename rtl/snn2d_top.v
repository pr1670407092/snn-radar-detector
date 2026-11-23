`timescale 1ns/1ps
// Top level for the 32x32 time-frequency 2D RSN accelerator.
//
// 1024-byte TFG -> sigma-delta encoder -> six 2D conv/LIF/SEW layers -> global
// max -> int8 FC 32->2.  The engine is shared by all layers (runtime config);
// the membrane memory is 64 lanes x 24 bit wide at each group address.
module snn2d_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire        feat_we,
    input  wire [9:0]  feat_addr,
    input  wire [7:0]  feat_data,
    output reg         busy,
    output reg         done,
    output reg         decision,
    output reg signed [47:0] score0,
    output reg signed [47:0] score1,
    output wire [2:0]  dbg_layer,
    output wire [1:0]  dbg_t,
    output wire [3:0]  dbg_state,
    output reg  [16:0] dbg_count0,
    output reg  [16:0] dbg_count1,
    output reg  [16:0] dbg_count2,
    output reg  [16:0] dbg_count3,
    output reg  [16:0] dbg_count4,
    output reg  [16:0] dbg_count5
);

    // ---------------- memories ----------------
    reg [255:0]  w_mem  [0:1016];
    reg [23:0]   b_mem  [0:191];
    reg [23:0]   th_mem [0:5];
    reg [255:0]  fc_w_mem [0:1];
    reg [47:0]   fc_b_mem [0:1];
    reg [31:0]   spk_mem [0:3103];
    reg [1535:0] v_mem   [0:1151];

    initial begin
        $readmemh("w2d.mem",  w_mem);
        $readmemh("b2d.mem",  b_mem);
        $readmemh("th2d.mem", th_mem);
        $readmemh("fc_w2d.mem", fc_w_mem);
        $readmemh("fc_b2d.mem", fc_b_mem);
    end

    // FSM/readout state (declared early: used by the read-address mux/config)
    reg [3:0]  tstate;
    reg [1:0]  t;
    reg [2:0]  layer;
    reg [10:0] v6_addr;
    reg        rd_mode;

    // ---------------- encoder ----------------
    wire        enc_busy, enc_done, enc_clear_done;
    wire        enc_spk_we;
    wire [4:0]  enc_spk_row;
    wire [31:0] enc_spk_data;
    reg         enc_clear, enc_start;

    rate_encoder2d u_enc (
        .clk(clk), .rst(rst), .clear(enc_clear), .start(enc_start),
        .feat_we(feat_we), .feat_addr(feat_addr), .feat_data(feat_data),
        .busy(enc_busy), .done(enc_done), .clear_done(enc_clear_done),
        .spk_we(enc_spk_we), .spk_row(enc_spk_row), .spk_data(enc_spk_data)
    );

    // ---------------- 2D engine ----------------
    wire [10:0] w_addr, v_raddr, v_waddr;
    wire [7:0]  b_addr;
    wire [11:0] spk_raddr, spk_waddr;
    wire        eng_spk_we, eng_v_we;
    wire [31:0] eng_spk_wdata;
    wire [1535:0] eng_v_wdata;
    wire        eng_busy, eng_done;
    wire [16:0] eng_spk_count;
    reg         eng_start;

    reg [255:0]  w_data;
    reg [23:0]   b_data;
    reg [31:0]   spk_rdata;
    reg [1535:0] v_rdata;

    wire [10:0] v_raddr_m = rd_mode ? v6_addr : v_raddr;

    always @(posedge clk) begin
        w_data   <= w_mem[w_addr];
        b_data   <= b_mem[b_addr];
        spk_rdata <= spk_mem[spk_raddr];
        v_rdata  <= v_mem[v_raddr_m];
    end

    // runtime layer config
    reg [5:0]  cfg_cin, cfg_cout, cfg_hin, cfg_win, cfg_hout, cfg_wout;
    reg [1:0]  cfg_stride;
    reg        cfg_sew;
    reg [10:0] cfg_wbase, cfg_vbase;
    reg [7:0]  cfg_bbase;
    reg [11:0] cfg_spkin_base, cfg_spkout_base, cfg_sew_base;

    // registered layer config (removes layer -> LIF long path)
    reg [5:0]  cfg_cin_q, cfg_cout_q, cfg_hin_q, cfg_win_q, cfg_hout_q, cfg_wout_q;
    reg [1:0]  cfg_stride_q;
    reg        cfg_sew_q;
    reg [10:0] cfg_wbase_q, cfg_vbase_q;
    reg [7:0]  cfg_bbase_q;
    reg [11:0] cfg_spkin_base_q, cfg_spkout_base_q, cfg_sew_base_q;
    reg signed [23:0] cfg_th_q;

    always @* begin
        cfg_cin = 6'd1; cfg_cout = 6'd16;
        cfg_hin = 6'd32; cfg_win = 6'd32;
        cfg_hout = 6'd32; cfg_wout = 6'd32;
        cfg_stride = 2'd1; cfg_sew = 1'b0;
        cfg_wbase = 11'd0; cfg_bbase = 8'd0; cfg_vbase = 11'd0;
        cfg_spkin_base = 12'd0; cfg_spkout_base = 12'd32; cfg_sew_base = 12'd0;
        case (layer)
            3'd0: begin
                cfg_cin = 6'd1; cfg_cout = 6'd16;
                cfg_hin = 6'd32; cfg_win = 6'd32; cfg_hout = 6'd32; cfg_wout = 6'd32;
                cfg_stride = 2'd1; cfg_sew = 1'b0;
                cfg_wbase = 11'd0; cfg_bbase = 8'd0; cfg_vbase = 11'd0;
                cfg_spkin_base = 12'd0; cfg_spkout_base = 12'd32;
            end
            3'd1: begin
                cfg_cin = 6'd16; cfg_cout = 6'd16;
                cfg_hin = 6'd32; cfg_win = 6'd32; cfg_hout = 6'd32; cfg_wout = 6'd32;
                cfg_stride = 2'd1; cfg_sew = 1'b0;
                cfg_wbase = 11'd9; cfg_bbase = 8'd32; cfg_vbase = 11'd256;
                cfg_spkin_base = 12'd32; cfg_spkout_base = 12'd544;
            end
            3'd2: begin
                cfg_cin = 6'd16; cfg_cout = 6'd16;
                cfg_hin = 6'd32; cfg_win = 6'd32; cfg_hout = 6'd32; cfg_wout = 6'd32;
                cfg_stride = 2'd1; cfg_sew = 1'b1;
                cfg_wbase = 11'd153; cfg_bbase = 8'd64; cfg_vbase = 11'd512;
                cfg_spkin_base = 12'd544; cfg_spkout_base = 12'd1056;
                cfg_sew_base = 12'd32;
            end
            3'd3: begin
                cfg_cin = 6'd16; cfg_cout = 6'd32;
                cfg_hin = 6'd32; cfg_win = 6'd32; cfg_hout = 6'd16; cfg_wout = 6'd16;
                cfg_stride = 2'd2; cfg_sew = 1'b0;
                cfg_wbase = 11'd297; cfg_bbase = 8'd96; cfg_vbase = 11'd768;
                cfg_spkin_base = 12'd1056; cfg_spkout_base = 12'd1568;
            end
            3'd4: begin
                cfg_cin = 6'd32; cfg_cout = 6'd32;
                cfg_hin = 6'd16; cfg_win = 6'd16; cfg_hout = 6'd16; cfg_wout = 6'd16;
                cfg_stride = 2'd1; cfg_sew = 1'b0;
                cfg_wbase = 11'd441; cfg_bbase = 8'd128; cfg_vbase = 11'd896;
                cfg_spkin_base = 12'd1568; cfg_spkout_base = 12'd2080;
            end
            3'd5: begin
                cfg_cin = 6'd32; cfg_cout = 6'd32;
                cfg_hin = 6'd16; cfg_win = 6'd16; cfg_hout = 6'd16; cfg_wout = 6'd16;
                cfg_stride = 2'd1; cfg_sew = 1'b1;
                cfg_wbase = 11'd729; cfg_bbase = 8'd160; cfg_vbase = 11'd1024;
                cfg_spkin_base = 12'd2080; cfg_spkout_base = 12'd2592;
                cfg_sew_base = 12'd1568;
            end
        endcase
    end

    wire signed [23:0] cfg_th = $signed(th_mem[layer]);

    conv2d_engine u_eng (
        .clk(clk), .rst(rst), .start(eng_start),
        .busy(eng_busy), .done(eng_done),
        .cfg_cin(cfg_cin_q), .cfg_cout(cfg_cout_q),
        .cfg_hin(cfg_hin_q), .cfg_win(cfg_win_q),
        .cfg_hout(cfg_hout_q), .cfg_wout(cfg_wout_q),
        .cfg_stride(cfg_stride_q), .cfg_sew(cfg_sew_q),
        .cfg_wbase(cfg_wbase_q), .cfg_bbase(cfg_bbase_q), .cfg_vbase(cfg_vbase_q),
        .cfg_spkin_base(cfg_spkin_base_q), .cfg_spkout_base(cfg_spkout_base_q),
        .cfg_sew_base(cfg_sew_base_q), .cfg_th(cfg_th_q),
        .w_addr(w_addr), .w_data(w_data),
        .b_addr(b_addr), .b_data(b_data),
        .spk_raddr(spk_raddr), .spk_rdata(spk_rdata),
        .spk_we(eng_spk_we), .spk_waddr(spk_waddr), .spk_wdata(eng_spk_wdata),
        .v_raddr(v_raddr), .v_rdata(v_rdata),
        .v_we(eng_v_we), .v_waddr(v_waddr), .v_wdata(eng_v_wdata),
        .spk_count(eng_spk_count)
    );

    // ---------------- memory write muxes ----------------
    wire        spk_we   = enc_spk_we | eng_spk_we;
    wire [11:0] spk_waddr_m = enc_spk_we ? {7'd0, enc_spk_row} : spk_waddr;
    wire [31:0] spk_wdata_m = enc_spk_we ? enc_spk_data : eng_spk_wdata;
    always @(posedge clk) begin
        if (spk_we)
            spk_mem[spk_waddr_m] <= spk_wdata_m;
    end

    reg         clear_v_we;
    reg [10:0]  clear_v_addr;
    wire        v_we = clear_v_we | eng_v_we;
    wire [10:0] v_waddr_m = clear_v_we ? clear_v_addr : v_waddr;
    wire [1535:0] v_wdata_m = clear_v_we ? 1536'd0 : eng_v_wdata;
    always @(posedge clk) begin
        if (v_we)
            v_mem[v_waddr_m] <= v_wdata_m;
    end

    // ---------------- top FSM ----------------
    localparam T_IDLE      = 4'd0,
               T_CLEAR_V   = 4'd1,
               T_CLEAR_ENC = 4'd2,
               T_ENC_START = 4'd3,
               T_ENC_WAIT  = 4'd4,
               T_L_START   = 4'd5,
               T_L_WAIT    = 4'd6,
               T_L_NEXT    = 4'd7,
               T_RD_SET    = 4'd8,
               T_RD_WAIT   = 4'd9,
               T_RD_GET    = 4'd10,
               T_FC        = 4'd11,
               T_FC_FIN    = 4'd12,
               T_DONE      = 4'd13,
               T_CLEAR_ENC_WAIT = 4'd14,
               T_FC_ADD    = 4'd15;

    reg [10:0] clear_cnt;
    reg [6:0]  rd_idx;
    reg [4:0]  fc_k;
    reg signed [23:0] maxv [0:31];
    reg signed [47:0] acc0, acc1;
    reg signed [31:0] prod0, prod1;

    assign dbg_layer = layer;
    assign dbg_t = t;
    assign dbg_state = tstate;

    integer l, co;

    always @(posedge clk) begin
        if (rst) begin
            tstate <= T_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            decision <= 1'b0;
            score0 <= 48'sd0;
            score1 <= 48'sd0;
            enc_clear <= 1'b0;
            enc_start <= 1'b0;
            eng_start <= 1'b0;
            clear_v_we <= 1'b0;
            clear_v_addr <= 11'd0;
            clear_cnt <= 11'd0;
            t <= 2'd0;
            layer <= 3'd0;
            rd_idx <= 7'd0;
            fc_k <= 5'd0;
            v6_addr <= 11'd0;
            rd_mode <= 1'b0;
            dbg_count0 <= 17'd0; dbg_count1 <= 17'd0; dbg_count2 <= 17'd0;
            dbg_count3 <= 17'd0; dbg_count4 <= 17'd0; dbg_count5 <= 17'd0;
        end else begin
            enc_start <= 1'b0;
            eng_start <= 1'b0;
            case (tstate)
                T_IDLE: begin
                    busy <= 1'b0;
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        done <= 1'b0;
                        clear_cnt <= 11'd0;
                        clear_v_addr <= 11'd0;
                        clear_v_we <= 1'b1;
                        t <= 2'd0;
                        layer <= 3'd0;
                        rd_mode <= 1'b0;
                        dbg_count0 <= 17'd0; dbg_count1 <= 17'd0; dbg_count2 <= 17'd0;
                        dbg_count3 <= 17'd0; dbg_count4 <= 17'd0; dbg_count5 <= 17'd0;
                        tstate <= T_CLEAR_V;
                    end
                end

                T_CLEAR_V: begin
                    clear_v_we <= 1'b1;
                    clear_v_addr <= clear_cnt;
                    if (clear_cnt == 11'd1151) begin
                        clear_v_we <= 1'b0;
                        tstate <= T_CLEAR_ENC;
                    end else begin
                        clear_cnt <= clear_cnt + 11'd1;
                    end
                end

                T_CLEAR_ENC: begin
                    enc_clear <= 1'b1;
                    tstate <= T_CLEAR_ENC_WAIT;
                end
                T_CLEAR_ENC_WAIT: begin
                    enc_clear <= 1'b0;
                    if (enc_clear_done)
                        tstate <= T_ENC_START;
                end

                T_ENC_START: begin
                    enc_start <= 1'b1;
                    tstate <= T_ENC_WAIT;
                end
                T_ENC_WAIT: begin
                    if (enc_done)
                        tstate <= T_L_START;
                end

                T_L_START: begin
                    cfg_cin_q <= cfg_cin; cfg_cout_q <= cfg_cout;
                    cfg_hin_q <= cfg_hin; cfg_win_q <= cfg_win;
                    cfg_hout_q <= cfg_hout; cfg_wout_q <= cfg_wout;
                    cfg_stride_q <= cfg_stride; cfg_sew_q <= cfg_sew;
                    cfg_wbase_q <= cfg_wbase; cfg_bbase_q <= cfg_bbase;
                    cfg_vbase_q <= cfg_vbase;
                    cfg_spkin_base_q <= cfg_spkin_base;
                    cfg_spkout_base_q <= cfg_spkout_base;
                    cfg_sew_base_q <= cfg_sew_base;
                    cfg_th_q <= $signed(th_mem[layer]);
                    eng_start <= 1'b1;
                    tstate <= T_L_WAIT;
                end
                T_L_WAIT: begin
                    if (eng_done)
                        tstate <= T_L_NEXT;
                end
                T_L_NEXT: begin
                    case (layer)
                        3'd0: dbg_count0 <= dbg_count0 + eng_spk_count;
                        3'd1: dbg_count1 <= dbg_count1 + eng_spk_count;
                        3'd2: dbg_count2 <= dbg_count2 + eng_spk_count;
                        3'd3: dbg_count3 <= dbg_count3 + eng_spk_count;
                        3'd4: dbg_count4 <= dbg_count4 + eng_spk_count;
                        3'd5: dbg_count5 <= dbg_count5 + eng_spk_count;
                    endcase
                    if (layer == 3'd5) begin
                        if (t == 2'd3) begin
                            for (l = 0; l < 32; l = l + 1)
                                maxv[l] <= -24'sd8388607;
                            rd_mode <= 1'b1;
                            rd_idx <= 7'd0;
                            tstate <= T_RD_SET;
                        end else begin
                            t <= t + 2'd1;
                            layer <= 3'd0;
                            tstate <= T_ENC_START;
                        end
                    end else begin
                        layer <= layer + 3'd1;
                        tstate <= T_L_START;
                    end
                end

                T_RD_SET: begin
                    v6_addr <= 11'd1024 + rd_idx;
                    tstate <= T_RD_WAIT;
                end
                T_RD_WAIT: begin
                    tstate <= T_RD_GET;
                end
                T_RD_GET: begin
                    for (l = 0; l < 64; l = l + 1) begin
                        co = l & 31;
                        if ($signed(v_rdata[24*l +: 24]) > maxv[co])
                            maxv[co] <= $signed(v_rdata[24*l +: 24]);
                    end
                    if (rd_idx == 7'd127) begin
                        acc0 <= 48'sd0;
                        acc1 <= 48'sd0;
                        fc_k <= 5'd0;
                        tstate <= T_FC;
                    end else begin
                        rd_idx <= rd_idx + 7'd1;
                        tstate <= T_RD_SET;
                    end
                end

                T_FC: begin
                    prod0 <= $signed(maxv[fc_k]) * $signed(fc_w_mem[0][8*fc_k +: 8]);
                    prod1 <= $signed(maxv[fc_k]) * $signed(fc_w_mem[1][8*fc_k +: 8]);
                    tstate <= T_FC_ADD;
                end
                T_FC_ADD: begin
                    acc0 <= acc0 + {{16{prod0[31]}}, prod0};
                    acc1 <= acc1 + {{16{prod1[31]}}, prod1};
                    if (fc_k == 5'd31)
                        tstate <= T_FC_FIN;
                    else begin
                        fc_k <= fc_k + 5'd1;
                        tstate <= T_FC;
                    end
                end
                T_FC_FIN: begin
                    score0 <= acc0 + $signed(fc_b_mem[0]);
                    score1 <= acc1 + $signed(fc_b_mem[1]);
                    decision <= (acc1 + $signed(fc_b_mem[1])) >
                                (acc0 + $signed(fc_b_mem[0]));
                    tstate <= T_DONE;
                end
                T_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    tstate <= T_IDLE;
                end
                default: tstate <= T_IDLE;
            endcase
        end
    end
endmodule
