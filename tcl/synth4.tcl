set root "G:/python-G/snn"
read_verilog [list \
  "$root/rtl/weight_mem.v" \
  "$root/rtl/rate_encoder.v" \
  "$root/rtl/lif_layer4.v" \
  "$root/rtl/snn_top4.v" ]
synth_design -top snn_top4 -part xc7z020clg400-2 -mode out_of_context
create_clock -name clk -period 20.000 [get_ports clk]
report_utilization -file util4_synth.rpt
report_utilization -hierarchical -file util4_synth_hier.rpt
report_timing_summary -file timing4_synth.rpt -max_paths 10
write_checkpoint -force post_synth4.dcp
puts "== SYNTH4 DONE =="
