# Timing constraints for board_top (PL-only self-test)
# 50 MHz board clock; MMCM derives 100 MHz for the core.
create_clock -name clk50 -period 20.000 [get_ports clk50]

# The MMCM output clock is derived automatically.  Keep the async input
# reset out of the timing graph.
set_false_path -from [get_ports rst_n]
