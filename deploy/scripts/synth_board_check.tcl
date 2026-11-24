# Synthesis-only board check (no pin constraints, no bitstream).
# Use this on a machine without the board to verify the board wrapper,
# MMCM inference, resources and timing before filling pins.xdc.
set root [file normalize [file join [file dirname [info script]] ..]]
set build $root/build/board_synth
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
synth_design -top board_top -part xc7z020clg400-2
report_utilization -file util_board_synth.rpt
report_timing_summary -file timing_board_synth.rpt -max_paths 10
write_checkpoint -force post_synth_board.dcp
puts "== BOARD SYNTH CHECK DONE =="
