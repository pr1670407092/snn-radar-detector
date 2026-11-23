set root "G:/python-G/snn"
read_verilog [list \
  "$root/rtl/conv2d_engine.v" \
  "$root/rtl/rate_encoder2d.v" \
  "$root/rtl/snn2d_top.v" ]
synth_design -top snn2d_top -part xc7z020clg400-2 -mode out_of_context
create_clock -name clk -period 20.000 [get_ports clk]
report_utilization -file util2d50_synth.rpt
report_utilization -hierarchical -file util2d50_synth_hier.rpt
report_timing_summary -file timing2d50_synth.rpt -max_paths 10
write_checkpoint -force post_synth_2d50.dcp
puts "== SYNTH2D50 DONE =="
