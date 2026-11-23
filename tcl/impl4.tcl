open_checkpoint post_synth4.dcp
opt_design
place_design
route_design
report_utilization -file util4_impl.rpt
report_utilization -hierarchical -file util4_impl_hier.rpt
report_timing_summary -file timing4_impl.rpt -max_paths 10
report_power -file power4_impl.rpt
write_checkpoint -force post_route4.dcp
puts "== IMPL4 DONE =="
