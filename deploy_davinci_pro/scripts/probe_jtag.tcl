# 只读探测：列出 JTAG 上可见的器件（不下载、不改板子状态）
open_hw_manager
connect_hw_server -allow_non_jtag

# get_hw_targets 在探测不到目标时会「抛异常」，不是返回空列表，
# 所以必须用 catch 包住，否则连提示都打印不出来。
if {[catch {set tgs [get_hw_targets]} err]} {
    puts "== NO JTAG TARGET =="
    puts "  ($err)"
    puts "检查：1) USB 下载器是否插好（设备管理器应出现 USB Serial Converter A/B 或 Digilent USB Device）"
    puts "      2) 板子是否上电、JTAG 排线 Pin1 是否对正"
    disconnect_hw_server
    close_hw_manager
    exit 1
}
if {[llength $tgs] == 0} {
    puts "== NO JTAG TARGET (empty list) =="
    disconnect_hw_server
    close_hw_manager
    exit 1
}

puts "PROBE_TARGETS: $tgs"
foreach t $tgs {
    current_hw_target $t
    if {[catch {open_hw_target} err]} {
        puts "PROBE_OPEN_FAIL: $t : $err"
        continue
    }
    foreach d [get_hw_devices] {
        puts "PROBE_DEVICE: $d  PART=[get_property PART $d]  IDCODE=[get_property IDCODE $d]"
    }
    close_hw_target
}
disconnect_hw_server
close_hw_manager
puts "== PROBE DONE =="
