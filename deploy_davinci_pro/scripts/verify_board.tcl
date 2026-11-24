# 回读配置状态：确认 FPGA 的 DONE 位真的起来了
open_hw_manager
connect_hw_server -allow_non_jtag
set t [lindex [get_hw_targets] 0]
current_hw_target $t
open_hw_target
set d [lindex [get_hw_devices] 0]
current_hw_device $d

foreach prop {REGISTER.IR REGISTER.IR.BIT5_DONE REGISTER.IR.BIT12_INIT_B REGISTER.CONFIG_STATUS} {
    if {[catch {set v [get_property $prop $d]} err]} {
        puts "PROP $prop : <n/a>"
    } else {
        puts "PROP $prop : $v"
    }
}
puts "PART: [get_property PART $d]"

# 若设计里有 ILA，这里会列出来；本设计没有 ILA，故应为空
if {[catch {set ilas [get_hw_ilas -quiet]} e]} { set ilas "" }
puts "ILAS: $ilas"

close_hw_target
disconnect_hw_server
close_hw_manager
puts "== VERIFY DONE =="
