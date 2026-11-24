# Build the PL-only board self-test bitstream.
# Usage (Windows):
#   G:\app\vivado2025.2\2025.2\Vivado\bin\vivado.bat -mode batch ^
#     -source deploy\scripts\build_board.tcl
#
# Before running:
#   1) copy deploy/constraints/pins.xdc.template to deploy/constraints/pins.xdc
#   2) replace every <...> with the real 领航者 ZYNQ7020 pin
#   3) check IOSTANDARD / LED polarity

set root [file normalize [file join [file dirname [info script]] ..]]
set build $root/build/board
set pins  $root/constraints/pins.xdc
set bit   $build/board_top.bit

if {![file exists $pins]} {
    error "Missing $pins. Copy constraints/pins.xdc.template to pins.xdc and fill the real board pins first."
}
set fh [open $pins r]; set ptxt [read $fh]; close $fh
if {[regexp {PACKAGE_PIN[[:space:]]*<} $ptxt]} {
    error "pins.xdc still contains <...> placeholders. Fill the real PACKAGE_PIN values first."
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

synth_design -top board_top -part xc7z020clg400-2
opt_design
place_design
route_design
write_bitstream -force $bit

report_utilization -file util_board.rpt
report_timing_summary -file timing_board.rpt -max_paths 10
report_power -file power_board.rpt
puts "== BOARD BITSTREAM DONE: $bit =="
