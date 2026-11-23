open_checkpoint post_synth_eff.dcp
opt_design
place_design
route_design
report_utilization -file util_eff_impl.rpt
report_utilization -hierarchical -file util_eff_impl_hier.rpt
report_timing_summary -file timing_eff_impl.rpt -max_paths 10
report_power -file power_eff_impl.rpt
write_checkpoint -force post_route_eff.dcp
puts "== IMPLEFF DONE =="
