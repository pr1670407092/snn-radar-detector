# Core-only OOC build (no board pins).  Useful for resource/timing regression.
set root [file normalize [file join [file dirname [info script]] ..]]
set build $root/build/core
file mkdir $build
cd $build
foreach f {w2d.mem b2d.mem th2d.mem fc_w2d.mem fc_b2d.mem} {
    file copy -force $root/data/$f $build/$f
}
read_verilog [list \
    $root/rtl/conv2d_engine.v \
    $root/rtl/rate_encoder2d.v \
    $root/rtl/snn2d_top.v ]
synth_design -top snn2d_top -part xc7z020clg400-2 -mode out_of_context
create_clock -name clk -period 10.000 [get_ports clk]
opt_design
place_design
route_design
report_utilization -file util_core.rpt
report_timing_summary -file timing_core.rpt -max_paths 10
write_checkpoint -force post_route_core.dcp
puts "== CORE BUILD DONE =="
