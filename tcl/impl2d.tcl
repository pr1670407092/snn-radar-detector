open_checkpoint post_synth_2d.dcp
opt_design
place_design
phys_opt_design
route_design
report_utilization -file util2d_impl.rpt
report_utilization -hierarchical -file util2d_impl_hier.rpt
report_timing_summary -file timing2d_impl.rpt -max_paths 10
report_power -file power2d_impl.rpt
write_checkpoint -force post_route_2d.dcp
puts "== IMPL2D DONE =="
