# Build the PL-only board self-test bitstream for 正点原子 达芬奇Pro (XC7A100T-FGG484).
#
# Usage:
#   set PROCESSOR_ARCHITECTURE=AMD64   (必要：WorkBuddy 的 shell 里这个变量是空的，
#                                       缺了 vivado.bat 会静默 exit 1)
#   F:\vivado\2025.2\Vivado\bin\vivado.bat -mode batch -source build_board.tcl
#
# 引脚已在 constraints/pins.xdc 里填好（R4 / U7 / W7 Y7 Y8 V9）。

set root  [file normalize [file join [file dirname [info script]] ..]]
set build $root/build/board
set pins  $root/constraints/pins.xdc
set bit   $build/board_top.bit
set part  xc7a200tfbg484-2

if {![file exists $pins]} {
    error "Missing $pins."
}
set fh [open $pins r]; set ptxt [read $fh]; close $fh
if {[regexp {PACKAGE_PIN[[:space:]]*<} $ptxt]} {
    error "pins.xdc still contains <...> placeholders."
}

file mkdir $build
cd $build
foreach f {w2d.mem b2d.mem th2d.mem fc_w2d.mem fc_b2d.mem feat_all.mem exp_dec.mem} {
    file copy -force $root/data/$f $build/$f
}

read_verilog [list \
    $root/rtl/conv2d_engine.v \
    $root/rtl/rate_encoder2d.v \
    $root/rtl/snn2d_top.v \
    $root/rtl/board_top.v ]
read_xdc $root/constraints/timing.xdc
read_xdc $pins

synth_design -top board_top -part $part
opt_design
place_design
route_design
write_bitstream -force $bit

report_utilization -file util_board.rpt
report_timing_summary -file timing_board.rpt -max_paths 10
report_power -file power_board.rpt
report_drc -file drc_board.rpt

puts "== BOARD BITSTREAM DONE: $bit =="
