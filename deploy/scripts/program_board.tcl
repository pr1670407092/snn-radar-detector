# Program the board self-test bitstream over JTAG.
# Usage: vivado -mode batch -source deploy/scripts/program_board.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
set bit  $root/build/board/board_top.bit
if {![file exists $bit]} { error "Missing $bit; run build_board.tcl first." }
open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices] 0]
current_hw_device $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
puts "== PROGRAMMED: $bit =="
