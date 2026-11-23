open_checkpoint post_synth.dcp
opt_design
place_design
route_design
report_utilization -file util_impl.rpt
report_utilization -hierarchical -file util_impl_hier.rpt
report_timing_summary -file timing_impl.rpt -max_paths 10
report_power -file power_impl.rpt
write_checkpoint -force post_route.dcp
puts "== IMPL DONE =="
