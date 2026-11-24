# 达芬奇Pro（正点原子 Artix-7 XC7A200T-FBG484）部署包

把 `snn-radar-detector-main/deploy`（原目标是**领航者 ZYNQ-7020**）移植到**正点原子 达芬奇Pro**。
纯 PL 设计：不用 PS、不用 DDR、不用 MIG，JTAG 下载后按复位键即可跑 8 例自检。

> **本板实测型号是 XC7A200T**（JTAG IDCODE `0x13636093`），对应 Vivado part `xc7a200tfbg484-2`。
> 见第 3 节。

> 📄 **要给老师看的验收材料在 [`验收材料/`](验收材料/) 目录**：
> `验收报告.html`（可打印成 PDF）、`验收报告.md`、`材料清单与提交说明.md`、`evidence/`（原始日志）。
> 其中「LED 实拍照片」需在板前自行补拍，方法见 `材料清单与提交说明.md` 第四节。

## 1. 底板资料（全网查证 + 板上实测）

达芬奇Pro = 正点原子 Artix-7 开发板（底板 + 可换核心板 35T/100T/200T）。
官方资料页：<https://www.openedv.com/docs/boards/fpga/zdyz_dafenqipro.html>（原理图/例程只在百度网盘，未直接取到）。

引脚号取自两处**相互独立、且都明确标注“达芬奇Pro / XC7A100T / FGG484”**的工程约束：

| 来源 | 内容 |
|---|---|
| gitee `zdevt/cortex_m3_on_xc7a100t`（README 的管脚约束） | clk=R4, rst_n=U7, led[3..0]=V9/Y8/Y7/W7, key[3..0]=T4/T3/R6/T6 |
| B站《蜂鸟E203移植到达芬奇PRO（XC7A100T）》的 xdc | CLK50MHZ=R4, mcu_rst=U7, LED=Y7/W7（V9/Y8 同组） |

对照该板丝印：**LED0=W7 / LED1=Y7 / LED2=Y8 / LED3=V9**。

**为什么 100T 的引脚能用到 200T 上**：FGG484 与 FBG484 是同一焊盘尺寸的两种焊球工艺代号
（FGG=无铅 / FBG=有铅），焊盘完全兼容，正点原子 200T 核心板用的就是 FBG484。实测这些引脚
在 `xc7a200tfbg484-2` 里全部合法（IOB 6/6 绑定成功、布局布线通过），所以 `pins.xdc` 无需改动。

（注意：老款“达芬奇”（A35T，非 Pro）底板引脚完全不同：clk=R4, rst=U2, led=R2/R3/V2/Y2，
两代是不同底板，不能混用。）

底板要点（用于配置）：

| 项目 | 值 | 说明 |
|---|---|---|
| FPGA | **XC7A200T-2FBG484I** | Vivado part：`xc7a200tfbg484-2`（**不是** fgg484，200T 无此封装） |
| 时钟 | 板载 50 MHz 有源晶振（Y1，经缓冲器进 **R4**） | 与 `board_top.v` 的 MMCM 输入 20 ns 一致 |
| 复位 | RESET 按键，**默认高、按下为低**（低有效）→ **U7** | 直接接 `rst_n`，极性正确 |
| 用户 LED | 4 个，FPGA 输出**高电平点亮** | W7/Y7/Y8/V9 |
| PL 用户 bank 电压 | 3.3 V | 全部用 `LVCMOS33` |
| 配置 | `CFGBVS VCCO` / `CONFIG_VOLTAGE 3.3` | 已写进 `constraints/pins.xdc` |

## 2. 相对原包改了什么

| 文件 | 改动 |
|---|---|
| `constraints/pins.xdc` | 新增：把 `<...>` 占位符填成 R4 / U7 / W7 / Y7 / Y8 / V9，加 CFGBVS、UNUSEDPIN、压缩位流约束 |
| `scripts/build_board.tcl` | part 从 `xc7z020clg400-2` 改为 **`xc7a200tfbg484-2`**，增加 `report_drc` |
| `rtl/board_top.v` | 头注释改为达芬奇Pro；新增 `LED_ACTIVE_LOW`（默认 0=高点亮，若上板全反改成 1 重综合） |
| `scripts/program_board.tcl` | 打印 JTAG 器件型号/IDCODE，**自动比对位流器件规模**，不匹配就 SKIP 并说明修法 |
| `scripts/verify_board.tcl` | 新增：回读 `DONE` 位与配置状态，确认真的配置成功 |
| `一键上板.bat` | 双击 = 综合+实现+下载；`一键上板.bat program` 只下载 |

RTL 功能逻辑（6 层 2D conv/LIF/SEW、σ-Δ 编码器、全局 max + FC）**一字未改**，只改板级外壳。

### 2.1 ★ 修掉一个会导致上板永远失败的 bug（原工程遗留）

原 `board_top.v` 把期望判决 ROM 写成了 **1 位宽**：

```verilog
reg       exp_rom [0:7];        // 原写法：1 bit 宽
reg [7:0] exp_rom [0:7];        // 已改为 8 bit 宽
```

Vivado 综合对 1 位宽 memory 的 `$readmemh` 会**只报一条 warning 然后整条跳过**：

```
INFO:    [Synth 8-3876] $readmem data file 'feat_all.mem' is read successfully
WARNING: [Synth 8-2898] ignoring malformed $readmem task: invalid memory name   ← exp_dec.mem 被丢弃
```

后果：位流里 `exp_rom` 全 0，而期望值是 `1,1,1,1,0,0,0,0` → **前 4 例全判错、后 4 例判对
→ pass_count 恒为 4，LED3 永远不会亮**。

这个 bug 之所以长期潜伏，是因为**仿真器能正常处理 1 位 memory**，所以 `sim/` 里 8/8 全过，
而综合器不认——「仿真通过」完全不能证明 ROM 进了位流。原工程 README 也注明"没在真板上跑过"。

**验证方法（已执行）：**

| 检查项 | 修复前 | 修复后 |
|---|---|---|
| `Synth 8-3876` 成功条数 | 6 | **7**（含 exp_dec.mem） |
| `Synth 8-2898 malformed` | **1** | **0** |

> 排查口诀：编译日志里每一个 `$readmemh` 都必须有一条对应的 `8-3876 read successfully`，
> 出现任何 `8-2898 ignoring malformed` 都是真 bug，不是噪音。

### 2.2 加密验证：两条独立的仿真证据链

因为 LED 读数需要肉眼确认，另外搭了两层仿真，让"8/8 判决正确"这件事有可复现的机器证据：

| 层次 | 测试台 | 结果 |
|---|---|---|
| 核心 `snn2d_top` | 原工程 `sim/tb_snn2d_top.v` | `ALL 8 CASES PASS`（判决与 Python 参考一致） |
| 板级外壳 `board_top` | **本次新写** `sim_run/tb_board_top.v` | 46 ms 跑完 8 例；decisions=`1 1 1 1 0 0 0 0`；`pass_count=8`；`LED=1010` |

原工程**没有板级外壳的测试台**，而 ROM 寻址、自检 FSM、LED 映射恰恰是板级 bug 的高发区——
上面那个 1 位宽 ROM 的 bug 正属于这一层，只有给外壳写 TB 才可能发现。

数据一致性已核对（否则仿真结论不能代表上板）：

- `deploy/data/feat_all.mem` 逐段（每 1024 行）== `hw2d/feat_0..7.mem`；
- `deploy/data/exp_dec.mem` == `hw2d/expected_dec.mem`；
- `w2d/b2d/th2d.mem` == `weights2d/` 同名文件。

仿真目录：`G:\2026qimo\snn\sim_run\`（`run_sim.sh` 跑核心，`run_board_sim.sh` 跑外壳）。

## 3. 本机实测结果（Vivado 2025.2，已上板跑通）

**JTAG 实测到的真实器件是 XC7A200T，不是 100T。**
IDCODE `0x13636093`、`PART=xc7a200t`。200T **没有 fgg484 封装**（FGG=无铅焊球、FBG=有铅焊球，
两者是同一焊盘尺寸的两个封装代号），所以正确 part 是 `xc7a200tfbg484-2`（即 XC7A200T-2FBG484I）。
写 `xc7a200tfgg484-2` 会报 `[Common 17-162] Invalid option value specified for '-part'`。

```
Device : xc7a200tfbg484-2   (IDCODE 13636093)
LUT    :  28,890 / 133,800  (21.6%)
FF     :  64,443 / 267,600  (24.1%)
BRAM   :     100 / 365      (27.4%)
DSP    :       2 / 740
IOB    :       6 / 285      (clk50 + rst_n + led[3:0]，全部绑上，说明引脚在 FBG484 里合法)
WNS    : +0.182 ns      WHS : +0.058 ns     → 100 MHz 时序全过
DRC    : 0 个 critical，只有 2 条 DSP 流水线性能建议
ROM    : 7/7 个 $readmemh 全部成功预置（修复后）
产物    : build/board/board_top.bit  (4.22 MB, 2026-11-24 19:23)
```

**下载结果（已执行，两次均为修复前/修复后版本）：**

```
DEVICE: xc7a200t_0  PART=xc7a200t  IDCODE=00010011011000110110000010010011  (bitstream: a200t)
== PROGRAMMED: build/board/board_top.bit -> xc7a200t_0 ==
回读 REGISTER.IR.BIT5_DONE = 1     ← FPGA 已真正完成配置（不是"命令没报错"）
```

200T 上 LUT/FF/BRAM 都只用掉 22~27%，余量很大。

> 参考：原包目标是 xc7z020（领航者 ZYNQ），本包同一套 RTL 在 xc7a100tfgg484-2 上也跑通过
> （LUT 45.6%、BRAM 74.1%、WNS +0.379 ns）。**当前 build 目录里是 200T 的位流。**

## 4. 上板步骤

### 4.1 推荐：双击 `一键上板.bat`

| 用法 | 作用 |
|---|---|
| 双击（无参数） | 综合+实现+下载，完整流程 |
| `一键上板.bat program` | 只下载（位流已存在时用，几十秒） |
| `一键上板.bat verify` | 只回读 `DONE` 位，确认 FPGA 配置成功 |
| `一键上板.bat probe` | 只探测 JTAG 上有哪些器件 |

批处理内已处理两个环境陷阱：构建阶段把 `TEMP`/`TMP` 指向 `G:\vivado_tmp`（避开 C 盘不足导致的
静默挂死），而下载/回读阶段**清空**该变量（指向 G: 会让 `connect_hw_server` 失败，见 7.1）。

### 4.2 命令行

```bat
REM 必须先设 PROCESSOR_ARCHITECTURE，否则 vivado.bat 静默退出、不留日志
set PROCESSOR_ARCHITECTURE=AMD64

REM 构建（临时目录务必指到空间充足的盘）
set TEMP=G:\vivado_tmp
set TMP=G:\vivado_tmp
F:\vivado\2025.2\Vivado\bin\vivado.bat -mode batch -source scripts\build_board.tcl

REM 下载 + 回读（这两步要用默认临时目录）
set TEMP=
set TMP=
F:\vivado\2025.2\Vivado\bin\vivado.bat -mode batch -source scripts\program_board.tcl
F:\vivado\2025.2\Vivado\bin\vivado.bat -mode batch -source scripts\verify_board.tcl
```

下载后按板上 **RESET** 键重跑 8 例（下方第 5 节有完整判读表）。

## 5. 上板后要看什么（唯一无法从 JTAG 自动判定的事）

`DONE=1` 只能证明**配置成功**，不能证明 8 个测试用例判对——那要看板上 LED。
请按板上 **RESET** 键，然后观察 4 个 LED：

| LED | 引脚 | 含义 | 正常表现 |
|---|---|---|---|
| LED0 | W7 | 核 busy | 复位后亮约 47 ms 后灭（8×5.82 ms + 特征装载） |
| LED1 | Y7 | 8 个 case 跑完 | 常亮 |
| LED2 | Y8 | 最后一个 case 的判决 | **灭**（case7 是负例，期望 0） |
| LED3 | V9 | pass_count == 8 | **亮** = 8/8 全对 |

判读：

| 现象 | 含义 | 下一步 |
|---|---|---|
| LED1 亮 + **LED3 亮** | 8/8 全对 | 部署完成 |
| LED1 亮 + LED3 灭 | 跑完了但没全对 | 加点 ILA 抓 `done/decision/dbg_count*` |
| LED1 不亮 | 没跑完 / 时钟没起来 | 先查 R4 是否有 50 MHz、MMCM `locked`、U7 复位极性 |
| 4 个 LED 状态与预期全反 | LED 是低电平点亮 | `board_top.v` 里 `LED_ACTIVE_LOW` 改 1，重跑（约 40 min） |
| 全灭/全亮不响应 | 丝印映射与网上资料不同 | 按你自己板子的原理图改 `pins.xdc` 的 4 个 `PACKAGE_PIN`，重跑 |

## 6. 已知限制

1. **35T 装不下**（20,800 LUT / 41,600 FF / 50 BRAM，本设计要 28.9K / 64.4K / 100）。
2. **LED 引脚号无官方 PDF 背书**：来自两处独立工程约束交叉确认，且 100T/200T 焊盘兼容使其可迁移；
   但正点原子 200T 核心板的实际走线未取得官方原理图核验，所以第 5 节的 LED 判读是最后一道关。
3. **无 ILA**：当前位流里没有调试核，出错时只能靠 LED。需要时把 ILA 挂到 `u_core` 的
   `start/busy/done/decision/dbg_layer/dbg_t/dbg_state/dbg_count0..5` 再重建。

## 7. 环境备忘（本机）

- Vivado 实际装在 **`F:\vivado\2025.2\Vivado\bin\vivado.bat`**（不在项目 README 写的 `G:\app` 下）。
- 从命令行调 `vivado.bat` **必须先设 `PROCESSOR_ARCHITECTURE=AMD64`**，否则 `loader.bat` 判定
  "Unsupported architecture" 后静默 `exit /b 1`，**不产生任何日志**，极易误判为综合失败。
- 200T 全流程约 **17~65 分钟**（视临时目录所在盘而定），综合阶段约 7 分钟。
- 当前 JTAG 已连通：`xc7a200t_0`，IDCODE `13636093`。

### 7.1 ★ 综合会在 C 盘空间不足时静默挂死

本机 **C 盘只剩约 4 GB**，而 `TEMP`/`TMP` 默认指向 `C:\Users\lx\AppData\Local\Temp`。
实现阶段（place/route）临时文件很大，空间不足时 Vivado **不报错、不退出，只是挂住**：

| 症状 | 判据 |
|---|---|
| 日志停在 `Synthesis finished` 之后再无输出 | 日志行数 30 分钟不增长 |
| 进程还在，但 CPU 时间几乎不涨 | 两次采样 CPU 只差 ~0.7 s |
| 中间文件时间戳冻结 | `build/*/.Xil/**` 全部停在同一个时刻 |

排查命令：

```powershell
Get-Process vivado | Select Id,CPU,WS
Get-PSDrive C,F,G | Select Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB)}}
```

**解决：把临时目录指到空间充足的盘再构建**（本机 G 盘 ~253 GB 空闲）：

```bash
TMP=G:/vivado_tmp TEMP=G:/vivado_tmp XILINX_TEMP=G:/vivado_tmp \
  F:/vivado/2025.2/Vivado/bin/vivado.bat -mode batch -source scripts/build_board.tcl
```

> 注意：**只对构建生效**。编程/回读这类轻量操作反而要用默认临时目录——把 `TMP` 改到
> `G:\` 后 `connect_hw_server` 会连不上 `cs_server`（`Labtools 27-3733`），换回默认即可。
> 另外强杀过 Vivado 之后，`hw_server` 可能残留占端口，先清进程再重试。
