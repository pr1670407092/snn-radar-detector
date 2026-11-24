# ============================================================
# 正点原子 达芬奇Pro (Artix-7 XC7A100T-FGG484) 引脚约束
# ============================================================
# 引脚来源（两处独立资料一致，均明确标注“达芬奇Pro / XC7A100T / FGG484”）：
#   1) gitee.com/zdevt/cortex_m3_on_xc7a100t  README 中的管脚约束
#      clk=R4  rst_n=U7  led[3..0]=V9/Y8/Y7/W7  key[3..0]=T4/T3/R6/T6
#   2) B站《蜂鸟E203移植到达芬奇PRO（XC7A100T）》中的 xdc
#      CLK50MHZ=R4  mcu_rst=U7  LED=Y7/W7（V9/Y8 亦属同组 LED）  KEY=R6/T4/T3
# 与该板 4 个 LED 的丝印对应关系（正点原子原理图 LED0..LED3）：
#   LED0=W7  LED1=Y7  LED2=Y8  LED3=V9
# PL 用户 bank 为 3.3V，LED 为高电平点亮（FPGA 输出高 => 亮），
# KEY 默认高、按下为低 => rst_n 低有效，与 board_top.v 一致。
#
# 若上板后发现 LED 全反（该亮的不亮），把 board_top.v 里的 LED_ACTIVE_LOW 改成 1 即可。

# 50 MHz PL clock  (板载 Y1 晶振，经缓冲器进 R4)
set_property -dict {PACKAGE_PIN R4 IOSTANDARD LVCMOS33} [get_ports clk50]

# active-low reset (板载 RESET/KEY 按键，按下为低)
set_property -dict {PACKAGE_PIN U7 IOSTANDARD LVCMOS33} [get_ports rst_n]

# LEDs: [0] busy, [1] all 8 done, [2] last decision, [3] pass_count==8
set_property -dict {PACKAGE_PIN W7 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN Y7 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN Y8 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN V9 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

# ---- 配置相关（Artix-7 上板必须）----
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
# 未使用引脚下拉，避免扩展口悬空
set_property BITSTREAM.CONFIG.UNUSEDPIN Pulldown [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
