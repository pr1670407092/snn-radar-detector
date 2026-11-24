# 领航者 ZYNQ-7020 板级部署包（PL-only 自检）

本目录把当前 2D 时频 RSN 加速器打包成**无 PS/DDR、插板即可自检**的部署包。宿主电脑没有板子也可以先做综合检查；到有板子的电脑上填好引脚后，一条 Tcl 生成 bitstream，再用 JTAG 下载。

## 1. 目录

```
deploy/
├── rtl/
│   ├── conv2d_engine.v       # 128-lane 2D conv + LIF + SEW 引擎（均衡版）
│   ├── rate_encoder2d.v      # 1024 通道 σ-Δ 编码器
│   ├── snn2d_top.v           # 6 层 × T=4 调度、v_mem/spk_mem、全局 max + FC
│   └── board_top.v           # 50→100 MHz MMCM、8 例自检 FSM、LED
├── data/
│   ├── w2d.mem / b2d.mem / th2d.mem / fc_w2d.mem / fc_b2d.mem
│   ├── feat_all.mem          # 8×1024 字节测试特征（连续存放）
│   ├── exp_dec.mem           # 8 个期望判决：1,1,1,1,0,0,0,0
│   └── expected_*.mem       # 参考分数/每层计数（可选，供 ILA/离线对比）
├── constraints/
│   ├── timing.xdc            # 50 MHz 输入时钟；MMCM 自动派生 100 MHz
│   └── pins.xdc.template     # ★ 必须按领航者原理图填写
├── scripts/
│   ├── synth_board_check.tcl # 无引脚，仅综合检查（无板电脑用）
│   ├── build_board.tcl       # 生成 board_top.bit（需 pins.xdc）
│   ├── build_core.tcl        # 只构建 snn2d_top 核（OOC 回归）
│   └── program_board.tcl     # JTAG 下载
└── build/                    # 生成物（.bit/.dcp/.rpt，已 gitignore）
```

## 2. 板级顶层做什么

`board_top.v`：

1. 输入 **50 MHz** 板载 PL 时钟，`MMCME2_BASE` 倍频到 **100 MHz**（VCO 1000 MHz，CLKOUT0 /10）。
2. 复位同步后，从 `feat_all.mem` 依次取 8 个 1024 字节测试向量，写入 `snn2d_top` 的特征端口。
3. 每个 case 拉高 `start`，等待 `done`，把 `decision` 与 `exp_dec.mem` 比较，累计 `pass_count`。
4. 4 个 LED：

| LED | 含义 |
|---|---|
| `led[0]` | 核 busy（推理中） |
| `led[1]` | 8 个 case 全部跑完 |
| `led[2]` | 最后一个 case 的 decision（case7 是负例，正常为 0/灭） |
| `led[3]` | `pass_count == 8`，即 8/8 全对 |

正常现象：按复位后 `led[0]` 亮一小段时间（8×约 5.82 ms，加上特征装载，远小于 1 s），然后 `led[1]`、`led[3]` 亮，`led[2]` 灭。若 `led[3]` 不亮，说明自检未全过，先查时钟/复位/引脚，再用 ILA 看 `u_core.done/decision`。

## 3. 上板前必须做：填引脚

`constraints/pins.xdc.template` **不能直接用于 bitstream**。按你手上领航者 ZYNQ7020 的原理图/用户手册，替换：

- `clk50`：PL 端 50 MHz 时钟引脚；
- `rst_n`：复位按键（低有效；若是高有效，在 `board_top.v` 中取反）；
- `led[0..3]`：4 个 PL LED；
- `IOSTANDARD`：按 bank 电压改（多数 3.3 V 用 `LVCMOS33`）。

然后：

```bash
cp deploy/constraints/pins.xdc.template deploy/constraints/pins.xdc
# 编辑 pins.xdc，替换所有 <...>
```

`build_board.tcl` 会检查 `pins.xdc` 是否还有 `<`，有则拒绝生成，避免生成“引脚随机”的废 bitstream。

> 如果板子 PL 时钟不是 50 MHz：改 `board_top.v` 的 `CLKIN1_PERIOD`、`CLKFBOUT_MULT_F`、`CLKOUT0_DIVIDE_F`，并改 `timing.xdc` 的 `create_clock` 周期。核心必须跑 100 MHz。

## 4. 构建

### 4.1 无板电脑：先做综合检查

```bat
cd /d G:\python-G\snn
G:\app\vivado2025.2\2025.2\Vivado\bin\vivado.bat -mode batch ^
  -source deploy\scripts\synth_board_check.tcl
```

生成：

- `deploy/build/board_synth/util_board_synth.rpt`
- `deploy/build/board_synth/timing_board_synth.rpt`
- `deploy/build/board_synth/post_synth_board.dcp`

这一步能验证 MMCM、自检 FSM、ROM 初始化、端口/语法；不需要引脚。当前实测结果：
LUT **30,871（58.0%）**、FF **64,438（60.6%）**、BRAM **100（71.4%）**、LUT/FF 逻辑综合 WNS **+2.188 ns**（board wrapper，未加引脚；以你的实现报告为准）。

### 4.2 有板电脑：生成 bitstream

填好 `pins.xdc` 后：

```bat
G:\app\vivado2025.2\2025.2\Vivado\bin\vivado.bat -mode batch ^
  -source deploy\scripts\build_board.tcl
```

生成：

- `deploy/build/board/board_top.bit`
- `util_board.rpt` / `timing_board.rpt` / `power_board.rpt`

### 4.3 只回归核心

```bat
G:\app\vivado2025.2\2025.2\Vivado\bin\vivado.bat -mode batch ^
  -source deploy\scripts\build_core.tcl
```

## 5. 下载与运行

板子接 JTAG，Vivado 能识别 `xc7z020` 后：

```bat
G:\app\vivado2025.2\2025.2\Vivado\bin\vivado.bat -mode batch ^
  -source deploy\scripts\program_board.tcl
```

按板上复位键重跑 8 例。若需要看内部信号，在 `board_top.v` 的 `u_core` 上添加 ILA：

- `start/busy/done/decision`
- `dbg_layer/dbg_t/dbg_state`
- `dbg_count0..5`
- 可选 `score0/score1`（48-bit，ILA 位宽较大）

## 6. 资源与时序预期

核心均衡版（100 MHz OOC 实现）：

| 指标 | 结果 |
|---|---:|
| LUT | 29,048（54.6%） |
| FF | 64,633（60.8%） |
| BRAM | 98/140（70%） |
| WNS | +0.173 ns |
| 单次 | 5.82 ms |
| 判决 | 8/8 |

`board_top` 额外加 8 KB 特征 ROM、期望判决 ROM、自检 FSM、MMCM，预计 BRAM +2～3、LUT/FF 少量增加；以 `util_board_synth.rpt` 为准。50 MHz→100 MHz 的 MMCM 和 I/O 约束可能让 WNS 与 OOC 略有差异，**必须看 `timing_board.rpt`**；若 WNS 为负，先检查时钟约束/引脚，再考虑把核心降到 90 MHz 或加流水。

## 7. 已知边界

- 本包**没有在真实板子上跑过**（当前开发机无板），所以引脚、LED 极性、复位极性必须按你的板子核对。
- 只用 PL；不配置 PS/DDR，不需要 Petalinux/SDK。
- 8 个测试向量来自 `hw2d/`，期望判决 1,1,1,1,0,0,0,0；RTL 与 Python 整数参考不是全用例 bit-exact（L1 t0 位一致，跨 t 行缓冲仍有边角差异），但 8/8 判决正确。
- 功耗是 Vivado vectorless 估算（均衡版约 0.992 W），不是板上实测。
- 上板后若 `led[3]` 不亮：先确认 `clk50` 频率/引脚、`locked`、复位极性、LED 极性；再抓 `done/decision`。
