# Program the 达芬奇Pro self-test bitstream over JTAG.
#   F:\vivado\2025.2\Vivado\bin\vivado.bat -mode batch -source program_board.tcl
#
# 会先打印 JTAG 上所有器件（型号 / IDCODE），并核对 bitstream 的器件型号，
# 不匹配就明确报出来而不是让 Vivado 抛 27-3303。
set root [file normalize [file join [file dirname [info script]] ..]]
set bit  $root/build/board/board_top.bit
if {![file exists $bit]} { error "Missing $bit; run build_board.tcl first." }

open_hw_manager
connect_hw_server -allow_non_jtag
if {[catch {set tgs [get_hw_targets]} err]} {
    puts "== NO JTAG TARGET =="
    puts "  ($err)"
    puts "检查：1) USB 下载器是否插在电脑上（设备管理器里应出现 USB Serial Converter A/B 或 Digilent USB Device）"
    puts "      2) 板子是否上电、JTAG 排线方向是否正确"
    puts "      3) 若用 Platform Cable USB II，装 Vivado 自带的 cable 驱动"
    exit 1
}
if {[llength $tgs] == 0} {
    puts "== NO JTAG TARGET (empty list) =="
    exit 1
}

# bitstream 头部记录的目标器件（字段被长度字节拆开，例如 "..7a200tfb<0x0b>g484"），
# 所以只提取规模数字：a200t / a100t / a35t
set fh [open $bit rb]; set head [read $fh 600]; close $fh
set bitpart "unknown"
if {[regexp {a([0-9]+)t} $head -> bitnum]} { set bitpart "a${bitnum}t" }

foreach t $tgs {
    current_hw_target $t
    open_hw_target
    foreach d [get_hw_devices] {
        set p [get_property PART $d]
        puts "DEVICE: $d  PART=$p  IDCODE=[get_property IDCODE $d]  (bitstream: $bitpart)"
        set devnum "?"
        regexp {a([0-9]+)t} $p -> devnum
        if {$bitpart eq "a${devnum}t"} {
            current_hw_device $d
            set_property PROGRAM.FILE $bit $d
            program_hw_devices $d
            refresh_hw_device -update_hw_probes false $d
            puts "== PROGRAMMED: $bit -> $d =="
            puts "== Press RESET on the board; LED3 (V9) on = 8/8 pass =="
        } else {
            puts "== SKIP: bitstream is $bitpart, board is $p =="
            puts "== Fix: set part in build_board.tcl to ${p}fbg484-2 style and rebuild =="
        }
    }
    close_hw_target
}
disconnect_hw_server
close_hw_manager
