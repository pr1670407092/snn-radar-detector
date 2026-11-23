set root "G:/python-G/snn"
read_verilog [list \
  "$root/rtl/weight_mem.v" \
  "$root/rtl/rate_encoder.v" \
  "$root/rtl/lif_layer.v" \
  "$root/rtl/snn_top_eff.v" ]
synth_design -top snn_top_eff -part xc7z020clg400-2 -mode out_of_context
create_clock -name clk -period 20.000 [get_ports clk]
report_utilization -file util_eff_synth.rpt
report_utilization -hierarchical -file util_eff_synth_hier.rpt
report_timing_summary -file timing_eff_synth.rpt -max_paths 10
write_checkpoint -force post_synth_eff.dcp
puts "== SYNTHEFF DONE =="
