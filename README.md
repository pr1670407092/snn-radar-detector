# SNN 雷达目标检测加速器（ZYNQ-7020 / 正点原子领航者）

> 1D 三层 MLP（低延迟、bit-exact）+ 2D 六层时频 RSN（99.72%、128-lane 100 MHz 2D 引擎）。

面向“雷达回波有无目标”的二分类任务，在 **XC7Z020-2CLG400** 上完成了算法、1D/2D RTL、综合实现、技术报告和独立调研。

## 核心结果（先看这里）

| 方向 | 结果 |
|---|---|
| **算法最佳：时频 RSN** | STFT 32×32 + 2D 残差 spiking CNN，T=4，28,146 参数，6 层卷积全 spiking；float **99.72%** / int8 **99.62%**，PD 99.67% / PFA 0.22%，AUC **0.9998**；PFA 1e-3 → PD 99.47% / PFA 0.05% |
| **2D RTL（最新）** | 128-lane / **100 MHz**；**8/8 决策**；WNS **+0.259 ns**、WHS +0.122、0 失败；LUT **27,948（52.5%）**、FF **15,393（14.5%）**、BRAM **98/140（70%）**、DSP 2、**0.661 W**；**5.01 ms/次**（64-lane 9.01 ms，**1.80×**；能量/次 5.26→3.31 mJ） |
| **低延迟：1D 8-bit** | 512→256→128→2，164,096 突触，**96.64%**，PD 94.67% / PFA 1.44%，**1.77 ms**，BRAM 36.5/140；RTL 与 Python **bit-exact 8/8** |
| **容量实验：4-bit** | 512→704→704→2，857,472 突触，96.50%，PFA 2.48%，4.755 ms，BRAM 118.5/140；仅容量实验 |

- 1D 三版 RTL（512-512-512 / 512-256-128 / 512-704-704）均与 Python 整数模型逐计数一致；2D RTL **8/8 决策正确**，但逐计数/FC 分数尚未全用例 bit-exact：case0 的 L1 精确、L2–L5 差 <0.2%、L6 约 1.6%，其余随机例因跨 t 行缓冲边角出现 1–14% 计数漂移；决策不受影响，bit-exact 是收尾项。
- 混合方案：MF + 1D CNN + 32-32-2 SNN 头 97.94%（SNN 只有 2,112 突触，不算 SNN 主力，见 1.3）。
- **独立调研报告（9 页）**：[docs/survey.html](docs/survey.html) / [docs/survey.pdf](docs/survey.pdf)（雷达 SNN、QCFS/PASCAL/SEW、7020 硬件、参考文献与证据边界）。
- **技术报告（15 页）**：[docs/report.html](docs/report.html) / [docs/report.pdf](docs/report.pdf)（含 9.9 算法、9.10 100 MHz RTL）。
- **GitHub**：<https://github.com/pr1670407092/snn-radar-detector>

> 状态：已完成数据合成、ANN/SNN 训练、整数转换、1D/2D RTL、Vivado xsim 仿真、综合/实现（out-of-context）。**尚未上板**（当前没有板卡）。

---

## 1. 关键结果

| 项目 | 结果 |
|---|---|
| 网络结构 | 512 → 512 → 512 → 2（3 个突触层） |
| 突触数 / 参数量 | 525,312 / 526,338 |
| 权重精度 | int8（-127..127） |
| 输入 | 512 点 FFT 幅度谱，uint8 |
| 时间步 | T = 16 |
| ANN 测试精度 | **96.71%** |
| SNN 测试精度 | **96.75%** |
| 检测率 PD / 虚警率 PFA | **95.00% / 1.54%** |
| AUC | **0.9931** |
| 单次推理周期 | 279,714 cycles |
| 延迟 @ 50 MHz | **5.59 ms/次（约 179 次/s）** |
| 突触吞吐 | 约 1.50 G 突触操作/s（峰值 1.6 G/s，94%） |
| LUT | 12,735 / 53,200（23.94%） |
| FF | 12,650 / 106,400（11.89%） |
| BRAM | **128 / 140（91.43%）**（127×RAMB36 + 2×RAMB18） |
| DSP | **0 / 220** |
| 时序 @ 50 MHz | WNS **+2.853 ns**，WHS +0.106 ns（OOC 实现） |
| 功耗（vectorless） | 0.375 W（动态 0.261 + 静态 0.114，仅估算） |
| 仿真结果 | **8/8 用例 PASS**，count0/count1/decision 与 Python 完全一致 |

测试集 8,000 例；混淆矩阵：TN=3976，FP=62，FN=198，TP=3764。

按 SNR 分档：

| SNR 区间 | 样本数 | 精度 |
|---|---:|---:|
| [-8, -5) dB | 1611 | 89.14% |
| [-5, -2) dB | 2369 | 97.59% |
| [-2, 1) dB | 2405 | 99.42% |
| [1, 4) dB | 1615 | 99.13% |

低 SNR 档是主要误差来源；数据为合成数据，换真实雷达数据需要重新训练/校准。

## 1.1 4-bit 容量版（实验，不推荐作为最终版本）

> 这一版是容量/速度实验：857k 突触、118.5/140 BRAM，但 SNN 96.50%、PFA 2.48%，不如 8-bit 基线/效率版。推荐版本见 1.2。

在 8-bit 基线之上，把权重压到 **4-bit**、并行度从 32 lane 提到 **64 lane**，
网络扩到 **512 → 704 → 704 → 2（857,472 突触，+63%）**，并已完成 RTL、位精确仿真、综合/实现。

| 项目 | 8-bit 基线 | 4-bit 容量版 |
|---|---|---|
| 网络 | 512-512-512-2 | **512-704-704-2** |
| 突触 | 525,312 | **857,472（+63%）** |
| 权重 / lane | int8 / 32 | **int4 / 64** |
| SNN 精度 | 96.75% | **96.50%** |
| PD / PFA | 95.00% / 1.54% | **95.46% / 2.48%** |
| AUC | 0.9931 | 0.9921 |
| 周期 / 延迟 @50 MHz | 279,714 / 5.594 ms | **237,762 / 4.755 ms** |
| 吞吐 | 178.8 次/s | **210.3 次/s** |
| LUT / FF | 23.94% / 11.89% | 26.11% / 12.39% |
| BRAM | 128 / 140 | **118.5 / 140** |
| WNS @50 MHz | +2.853 ns | **+3.795 ns** |
| 功耗（vectorless） | 0.375 W | 0.381 W |

- 4-bit 版硬件仿真 **8/8 PASS**，count0/count1/decision 与 Python 完全一致：
  `build/sim4/sim4.log`
- 文件：`rtl/lif_layer4.v`、`rtl/snn_top4.v`、`sim/tb_snn_top4.v`、`weights4/`、
  `build/synth4/`、`docs/metrics4.json`
- 代价：PFA 从 1.54% 升到 2.48%，精度 −0.25%；如果更看重低虚警，8-bit 基线更稳。
- 4-bit 版阈值 `θ = [39, 15, 84]`，权重 −7…7，膜电位仍 24-bit。

## 1.2 8-bit 效率版（推荐）

在 8-bit 512-512-512-2 基础上把宽度缩到 **512-256-128-2**：任务在 256-128 就饱和，
网络小 69%，精度几乎不变，但延迟/BRAM 大幅下降。这是目前**质量/效率最好的版本**。

| 项目 | 8-bit 基线 512-512-512 | 8-bit 效率版 512-256-128 | 4-bit 容量版 512-704-704 |
|---|---|---|---|
| 突触 | 525,312 | **164,096（−69%）** | 857,472（+63%） |
| 权重 / lane | int8 / 32 | int8 / 32 | int4 / 64 |
| SNN 精度 | 96.75% | **96.64%** | 96.50% |
| PD / PFA | 95.00% / 1.54% | **94.67% / 1.44%** | 95.46% / 2.48% |
| AUC | 0.9931 | **0.9943** | 0.9921 |
| 周期 / 延迟 @50 MHz | 279,714 / 5.594 ms | **88,610 / 1.772 ms** | 237,762 / 4.755 ms |
| 吞吐 | 178.8 次/s | **564.3 次/s** | 210.3 次/s |
| LUT / FF | 23.94% / 11.89% | 21.81% / 10.61% | 26.11% / 12.39% |
| BRAM | 128 / 140 | **36.5 / 140（26.1%）** | 118.5 / 140 |
| WNS @50 MHz | +2.853 ns | **+3.254 ns** | +3.795 ns |
| 功耗（vectorless） | 0.375 W | **0.279 W** | 0.381 W |

- 硬件仿真 **8/8 PASS**，count0/count1/decision 与 Python 完全一致：
  `build/sim_eff/sim_eff.log`
- 文件：`rtl/snn_top_eff.v`、`sim/tb_snn_top_eff.v`、`weights_eff/`、
  `build/sim_eff/`、`build/synth_eff/`、`docs/metrics_eff.json`
- T=32 选项：96.73% / PD 95.03% / PFA 1.61% / AUC 0.9945，延迟约 3.54 ms。
- 4-bit 小网络 256-128（Python 评估）：96.78% / PD 95.41% / PFA 1.88% / AUC 0.9942，
  约 0.95 ms、BRAM 约 25，尚未做 RTL；如果速度优先，可以下一步实现。
- **PFA 校准（DEC_THRESH）**：目标 PFA 1e-2 → 阈值 2 → test PD **92.38%**（实测 PFA 0.54%）；
  5e-3 → 3 → PD 89.78%（0.20%）；1e-3 → 5 → PD 84.07%（0.00%，8000 例 0 FP）。
  AUC 0.9943；count argmax 96.64%；最佳验证阈值 -1 → 96.75%。RTL 8/8，阈值不改变 8 个用例的判决。
- 本地 RSN-style 1D 残差 spiking CNN（membrane readout）：92.64%，PD 80.44% @ PFA 1e-2；
  膜电位搬到 8-bit MLP 上 AUC 仅 0.62；count-margin PFA 更好。
- 结论：8-bit 效率版用 31% 突触、26% BRAM、1/3 延迟，拿到与基线几乎相同的精度、
  更低的 PFA 和更高的 AUC；4-bit 大网络只作为容量实验保留。

## 1.3 MF + 1D CNN + SNN 头（混合，试验）

把**匹配滤波 range profile**作为 1D 输入：紧凑 1D CNN（max pool、无残差、9,682 参数）
先做特征提取，再取 32 维 global-max 特征接 **32-32-2 SNN 头**。

| 方案 | 精度 | PD / PFA | AUC | 参数 / 突触 | 说明 |
|---|---|---|---|---|---|
| MF + 1D CNN（ANN） | **98.01%** | 97.15% / 1.16% | 0.9959 | 9,682 参数 | CNN 前端非 spiking |
| CNN 32 特征 + SNN 32-32-2（T=32） | **97.94%** | 97.40% / 1.54% | **0.9960** | **2,112 SNN 突触** + 9,682 CNN | **混合，推荐** |
| CNN 32 特征 + SNN 32-32-2（T=16） | 97.88% | 97.50% / 1.76% | 0.9957 | 2,112 SNN 突触 | 更低延迟 |
| 纯 spiking CNN（朴素 rate 转换） | 85.04% | 85.23% / 15.16% | 0.9246 | 9,682 参数 | **不可直接用** |
| 8-bit 效率 MLP（纯 SNN） | 96.64% | 94.67% / 1.44% | 0.9943 | 164,096 突触 | 已实现 RTL 的 1D 纯 SNN |

- MF 提供相干增益，CNN 的 **global max pool** 提供平移不变性；去掉残差仍 98.01%，
  换成 average pool 只有 95.1%。
- SNN 头 T=32 与 CNN 的 98.01% 只差 0.07%，SNN 只有 **2,112 个突触**。
- **这不是纯 SNN**：CNN 前端（9,682 参数）不是 spiking；纯 spiking CNN 的朴素 rate
  转换只有 85.04%，需要 QCFS/直接训练 + 卷积引擎。如果老师要求纯 SNN，
  9.9 的 2D 时频 RSN 已到 99.72%（2D 卷积引擎尚未 RTL）；已实现 RTL 的推荐仍是
  8-bit 256-128（96.64%，1.77 ms）。
- 文件：`python/train_cnn.py`、`python/convert_cnn_snn.py`、`python/extract_cnn_feat.py`、
  `models/cnn_mf_max_r0.pt`、`models/ann_hybrid.pt`、`data/mf_cnn_feat.npz`、
  `docs/metrics_cnn_mf.json`、`docs/metrics_hybrid_snn.json`；
  MF 特征生成脚本在 `archive/mf_mlp_experiment/`。

## 1.4 SNN 主力的近期模型（2023–2026）与结论

如果要求 SNN 做主要计算，当前最现实的三条路线：

1. **2D 时频 RSN（本地已验证，99.72%）**：STFT 32×32 + 2D 残差 spiking CNN，
   T=4，28,146 参数，6 层卷积全部 spiking；需要 2D 卷积引擎（3×3 行缓冲、stride-2、
   BN 折叠、LIF/SEW）。精度最高，已落成 128 lane/100 MHz RTL（5.01 ms/次）；事件驱动
   估算约 0.79 ms。见 1.5/1.6。
2. **QCFS/PASCAL 转换的小型 1D spiking CNN**：SNN 自己做 6 层 conv，
   T=8，10k–50k 权重，需要 conv engine。文献：QCFS（ICLR 2022）、PASCAL（TMLR 2025）、
   SEW-ResNet（NeurIPS 2021）。注意：QCFS 必须用 per-layer λ、θ=λ、v(0)=θ/2、T=L；
   我们第一版 QCFS 原型没对齐这些，只有 51%，不能当结论。
3. **Spiking U-Net**：SNN 输出干净 range profile、做杂波抑制，约 20–25k 权重、T=8、
   10–15 BRAM；最符合“SNN 主力”，但需要 per-bin 标签/掩码。

其他近期模型：

- **P-SpikeSSM**（ICLR 2025，~250k，全 spiking SSM）：8-bit 可放，但 causal scan 对 FPGA 不友好。
- **SpikCommander**（AAAI 2026，0.19M SHD 96.41%）：T=100 太长；1.12M SSC 也偏大。
- **MRA-SNN**（2024，1D 振动，1.75M）：参数太大；需 4-bit/DDR。
- **SpikingMamba-1.3B / SpikeGPT 45M–216M**：放不进 7020；Mamba-Spike 的 backbone 也非 spiking。
- **Gamma-ray spectral SNN**（2025，49 neurons/66 synapses，2 mW）：最接近“1D 光谱二分类”的 edge SNN，但任务不同。
- **雷达 SNN**：海杂波 RSN（PD 0.8266 @ PFA 1e-3）、HRRP SNN、STT 98.99%、EMRFormer 0.31–0.53M T=1；
  SNN 主力在雷达上可行，但暂无 7020 实现。

本地现实：纯 spiking CNN 的 direct BPTT 85.5%（1D）、naive rate 转换 85.0%、QCFS 原型 51%；
在时频 RSN 之前，最佳纯 SNN 是 8-bit 256-128（96.64%，1.77 ms），最佳精度是混合方案
（97.94%，SNN 头 2,112 突触）。**1.5 的 2D 时频 RSN 已经把纯 SNN 提到 99.72%**，
2D 卷积引擎 RTL 已完成功能仿真（见 1.6）。

## 1.5 时频图 + 2D 残差 spiking CNN（time-frequency RSN，最新）

按 GRSL 2025 RSN 的思路：512 点回波 → 64 点 STFT（hop 16，29 帧，补到 32×32）
→ log + median/MAD → 8-bit 时频图；先训 2D 残差 CNN 作为 ANN 上界，再把
**全部 6 层卷积换成 spiking**（SEW-ADD 残差、BN、LIF、膜电位读出、direct BPTT），T=4。

| 方案 | 测试精度 | PD / PFA | AUC | 参数 | 说明 |
|---|---|---|---|---|---|
| 1D 纯 SNN MLP 512-256-128 | 96.64% | 94.67% / 1.44% | 0.9943 | 164,096 突触 | 已 RTL，1.77 ms |
| MF + 1D CNN + SNN 头 | 97.94% | 97.40% / 1.54% | 0.9960 | 9,682 + 2,112 | CNN 前端非 spiking |
| STFT + 2D 残差 CNN（ANN 上界） | 99.83% | — | — | 28,146 | ANN 原型 |
| **STFT + 2D 残差 spiking CNN（T=4）** | **99.72%** | **99.67% / 0.22%** | **0.9998** | **28,146** | **纯 SNN，6 层卷积全 spiking** |

- 与 ANN 上界只差 **0.11 个点**；混淆矩阵 TN=4029、FP=9、FN=13、TP=3949。
- SNR 分档：[-8,-5) 98.94%、[-5,-2) 99.92%、[-2,1) 99.96%、[1,4) 99.88%。
  1D 纯 SNN 的低 SNR 档只有 89.14%，时频 RSN 提升到 98.94%。
- PFA 校准（膜电位）：1e-2 → PD 99.82%（实测 PFA 1.31%）；5e-3 → 99.82% / 0.67%；
  1e-3 → 99.47% / 0.05%。
- 算力：每时间步 10.76M MAC，T=4 共 43.06M；实测各层输入脉冲率
  53.7% / 20.0% / 24.7% / 29.3% / 22.9% / 20.8%，事件驱动有效 MAC 约 10.05M。
  64 lane/100 MHz dense 6.73 ms（实测 9.01 ms）、128 lane dense 3.36 ms
  （实测 **5.01 ms**，见 1.6）；事件驱动 1.57/0.79 ms 为估算，尚未实现。
- 存储：膜电位 73,728×24 bit；实测 BRAM **98/140**（含权重/行缓冲），
  LUT 实测 **14,946（64 lane）/27,948（128 lane）**。
- 注意：STFT 是非 spiking 预处理；分类所需的 6 层卷积/残差/读出全部 spiking，
  属于“SNN 主力”。合成数据比 IPIX 简单，99.72% 不能直接和论文的 PD 0.8266 比。
- 文件：`python/tf_data.py`、`train_cnn2d.py`、`train_spiking_cnn2d.py`、
  `eval_spiking_cnn2d.py`、`models/cnn2d_tf.pt`、`models/spiking_cnn2d_tf.pt`、
  `data/radar_tf.npz`、`docs/metrics_spiking_cnn2d.json`。

## 1.6 2D 卷积引擎 RTL（128 lane / 100 MHz，功能仿真 8/8）

已把 1.5 的 2D SNN 写成可综合 RTL：

- `rtl/conv2d_engine.v`：128 lane 2D conv + LIF + SEW；3×3 三行 line buffer；
  运行时层配置（Cin 1/16/32、Cout 16/32、stride 1/2、SEW）；int8 权重、24-bit 膜电位；
  一个 tap/cycle；bias/residual/threshold 全部 signed。128 lane = **8 pixel×16 ch**
  或 **4 pixel×32 ch**，每行 **4 组**；行缓冲读出的 tap 窗口位先打一拍（`sg_r`），
  把长 mux 从 128 lane 的 acc 使能路径上移走。
- `rtl/rate_encoder2d.v`：1024 通道 sigma-delta，1 pixel/cycle，T=4，acc 跨 t 保持。
- `rtl/snn2d_top.v`：6 层调度、全局 max、int8 FC 32→2；v_mem **3072×576**（128 lane×24 bit，
  576 = 6 层膜电位地址空间），spk_mem 3104×32，编码器/层/FC FSM。
- `sim/tb_snn2d_top.v`：8 个测试向量（4 正/4 负），对比 decision、FC 分数和每层脉冲计数。

功能仿真结果：**8/8 decision PASS**；L1 t0 的输出图和膜电位 **0 bit 差**；case0 的
L2–L5 计数差 <0.2%、L6 约 1.6%，FC 分数差约 1.5%。其余随机例的逐层计数有 1–14%
漂移（跨 t 的 row-buffer 边角 + 编码器/计数观测时机），FC 分数也会漂移，但 8 例
判决全部正确。**2D RTL 尚未全用例 bit-exact**，这是后续收尾项；1D 三版 RTL 仍是
完全 bit-exact 的。

**深度/宽度消融（2026-11-23）：**

| 数据集 | 6 层 28k | 8 层 47k | width24 63k |
|---|---|---|---|
| 简单集（SNR −7…3，CNR 8…16） | **99.72%** | 99.49% | 99.61% |
| 难集（SNR −10…0，CNR 8…18，K 杂波/尖峰/干扰/缺失/失配） | **76.89%**（AUC 0.8441） | 76.98%（0.8397） | 76.68%（0.8429） |

难集 PFA 1e-2 时 PD：6L 46.7%、8L 46.1%、W24 48.0%。**加层/加宽在简单集和难集都没有明显收益**；简单集已饱和，难集才是真实瓶颈。下一步优先 T=8、特征/预处理、真实 IPIX；128-lane 已把单次压到 5.01 ms，继续堆层没有收益。

- 延迟：8 例仿真共 **40.07 ms** → 约 **5.01 ms/次 @100 MHz**（含特征写入、4 次编码、
  6 层 dense MAC、读出/FC）。64-lane 同版本为 9.01 ms，128-lane 实测 **1.80×**
  提速；dense MAC 理论 3.36 ms，固定开销约 1.65 ms；事件驱动约 0.79 ms 仍是估算。
- 资源（Vivado 2025.2，xc7z020clg400-2，OOC，100 MHz 实现）：**LUT 27,948/53,200（52.5%）**、
  **FF 15,393/106,400（14.5%）**、**BRAM 98/140（70%）**、**DSP 2/220**；功耗 **0.661 W**
  （动态 0.543 + 静态 0.118）。相比 64-lane：LUT 1.87×、FF 1.48×、BRAM 不变、功耗 1.13×，
  单次能量 5.26→**3.31 mJ**（−37%）。
- 时序：100 MHz 实现 **WNS +0.259 ns、WHS +0.122 ns，0 失败端点**（综合 WNS +2.373）。
  流水化：sg 预取（`sg_r`）→ 配置/阈值寄存 → pc/计数寄存 → tap 计数器（去 /9）
  → 两级 LIF（fire_vec 寄存）→ FC 乘/加两级。首版 128-lane 实现 WNS −0.712 ns，
  加 sg 预取后收敛到 +0.259。实现结果见 `build/synth2d/timing2d_impl.rpt`。
- 复现：`python/convert_snn2d.py` → 生成 `weights2d/`、`hw2d/`；`build/sim2d` 下 xvlog/xelab/xsim；
  `tcl/synth2d.tcl` / `tcl/impl2d.tcl`；仿真日志 `docs/sim2d_128.log`（8/8 PASS）。

---

## 2. 板卡与资源约束

- 板卡：正点原子 **领航者 ZYNQ7020**，器件 `xc7z020clg400-2`（-2 speed grade）。
- 7020 资源：53,200 LUT、106,400 FF、140×36 Kb BRAM、220 DSP。
- PL 时钟：50 MHz（板载晶振），未使用 PLL/MMCM。
- 综合/实现：Vivado 2025.2，out-of-context（无 I/O 约束，未上板）。

### 为什么是 512-512-512-2，而不是更大？

权重按 **每 32 个 int8 打包成 256 bit 字** 存储，行优先：

```
word_addr = dest_neuron * (FANIN/32) + chunk
```

BRAM 的深度会落在不同的原语模式上：

| 每层字数 | 深度模式 | 256 bit 宽所需 BRAM |
|---|---:|---:|
| 4096 | 4K×9 | 32 |
| **8192** | **8K×4** | **64** |
| 11,264 | 16K×2 | 128 |
| 16,384 | 16K×2 | 128 |

512 个隐藏神经元、fan-in 512 时，一层正好 **512×16 = 8192 字 → 64 个 RAMB36**。
最终实现中 L1 用 64 个 RAMB36、L2 用 63 个 RAMB36，加 2 个 RAMB18 的膜电位，
合计 **128/140 个 BRAM tile**。若第二层改成 704 神经元，一层就变成 11,264 字 →
128 个 BRAM，两层放不下。
因此 512-512-512-2 是在“int8 权重 + 256 bit 字 + 32 lane”组织下
**接近 BRAM 上限的最大网络**。若要继续增大，需要 4-bit 权重或权重压缩。

---

## 3. 硬件架构

```
雷达复基带回波
  │  512 点 FFT → 10log10|X|² → median/MAD 归一化 → uint8
  ▼
512 字节特征
  │  sigma-delta 速率编码（T=16，512 个 9-bit 累加器）
  ▼
每时间步 512-bit 脉冲向量
  │
  ├─ L1: 512 LIF，W1 512×512 int8，8192×256b BRAM
  ├─ L2: 512 LIF，W2 512×512 int8，8192×256b BRAM
  └─ L3:   2 LIF，W3   2×512 int8，LUTRAM
  ▼
16 个时间步的输出脉冲计数 → argmax → 有/无目标
```

### 3.1 高效 SNN 单元

- **32 路突触/cycle**：一个 256-bit 权重字 = 32 个 int8；每个 lane 做
  `p_i = spike_i ? w_i : 0`，再走 5 级 16-bit 加法树。
- **无乘法器**：脉冲是 0/1，突触操作退化为条件加；DSP 使用量为 0。
- **int8 权重 + 24-bit 有符号膜电位/偏置/阈值**；软复位：`V >= θ` 时 `V -= θ`。
- **泄漏可配置**（`LEAK_SHIFT`），当前为 0（IF 模式，转换精度最好）。
- **膜电位存 BRAM/LUTRAM**，避免 512 路 24-bit 读 mux 占用大量 LUT。
- **权重/偏置/膜电位预取**：地址组合驱动，读延迟 1 cycle，逐 chunk 流水。
- **S_UPDATE 流水级**：把“加法树 + 累加”和“膜电位更新 + 比较”分到两个 cycle，
  50 MHz 下把关键路径从约 19.6 ns 降到约 15.8 ns。
- **编码器控制寄存器复制**：512×9 个累加器的 clear/tick 扇出很大，改成寄存器驱动
  并加 `max_fanout` 约束，避免长路由。
- 推理开始前用写端口扫 0 清空膜电位；层间按 L1→L2→L3 顺序执行，状态跨时间步保留。

### 3.2 RTL 模块

| 文件 | 作用 |
|---|---|
| `rtl/snn_top.v` | 顶层 FSM、三层次序、输出计数/判决 |
| `rtl/lif_layer.v` | 通用 LIF 层：32-lane 条件加、权重/偏置/膜电位预取、S_UPDATE 流水 |
| `rtl/rate_encoder.v` | 512 路一阶 sigma-delta 速率编码 |
| `rtl/weight_mem.v` | 256-bit 宽权重 BRAM / LUTRAM，`$readmemh` 初始化 |
| `rtl/snn_params.vh` | 导出的 T/阈值，供文档核对 |

### 3.3 存储格式

- 权重文件 `w1.mem` / `w2.mem` / `w3.mem`：每行 64 个 hex digit（256 bit），
  行号 = `dest*W_WORDS + chunk`，lane 0 在 bit [7:0]。
- 偏置 `b1.mem` / `b2.mem` / `b3.mem`：每行 6 个 hex digit（24-bit 补码）。
- 特征 `hw_features.mem`：每行 1024 个 hex digit（512 字节，byte 0 在最低字节）。
- 期望结果 `hw_expected.mem`：每行 16 个 hex digit：
  `{count0[15:0], count1[15:0], decision}`（高位补 0）。

---

## 4. 数据与算法

### 4.1 合成雷达数据

复基带 LFM：

- `fs = 50 MHz`，`N = 512`，`Tp = 10 µs`，`B = 20 MHz`，`K = B/Tp`。
- 目标回波：延迟 0..160 samples、多普勒 ±80 kHz、随机相位、对数正态幅度；
  `SNR ∈ [-7, 3] dB`；无目标类别没有回波。
- 杂波：AR(1)（ρ∈[0.94,0.995]）+ 3 个 0.3..6 MHz 慢正弦；`CNR ∈ [8,16] dB`。
- 噪声：复高斯白噪声。

特征：512 点 FFT 的**全部 512 个 bin**（复基带正/负频率都保留），
`10log10(|X|²)`，逐样本中位数/MAD 鲁棒归一化，裁剪到 ±3σ，映射到 uint8。
使用完整 512 bin 比只用正频率 256 bin 精度从 92.8% 提升到 96.7%：
负频率半谱提供了额外的噪声/杂波参考，且不增加网络宽度（只增加输入维度）。

数据划分：24,000 训练 / 4,000 验证 / 8,000 测试，类别均衡，种子 2027。

### 4.2 ANN 训练

- 结构：512 → 512 → 512 → 2，ReLU，带 bias。
- 正则：dropout 0.2、weight decay 1e-2、输入高斯噪声 σ=0.03、label smoothing 0.05。
- 优化：AdamW，lr 1e-3，cosine，120 epochs，batch 256，seed 43。
- 结果：best val 96.83%，**test 96.71%**。

### 4.3 ANN → 整数 SNN

采用 rate coding + IF/LIF 软复位，逐层权重缩放：

1. 在 2,000 个校准样本上取每层激活的 **99.0 百分位** 作为 `λ_l`，`λ_0 = 1`。
2. `W'_l = W_l · λ_{l-1}/λ_l`，`b'_l = b_l/λ_l`。
3. 输出层两个 logit 加同一常数 `C = 2.487`（argmax 不变），再做 `λ_3` 缩放。
4. 整数量化：
   - `K_l = floor(127 / max|W'_l|)`
   - `W_q = clip(round(W'_l · K_l), -127, 127)`
   - `b_q = round(b'_l · K_l)`，阈值 `θ_l = K_l`
   - 第一层额外乘 `256/255`，补偿“编码器速率 q/256”与“ANN 输入 q/255”的差异。
5. 最终阈值：`θ = [316, 289, 3571]`；`λ = [0.3589, 0.3391, 4.5529]`。

时间步扫描（8,000 测试集）：

| T | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| SNN 精度 | 96.28% | **96.75%** | 96.78% | 96.79% |

T=16 与 T=32 精度几乎相同，但延迟减半，所以硬件取 **T=16**。

---

## 5. 仿真验证

Python 整数参考模型在 `python/convert_snn.py`：

- `encode()`：与 `rate_encoder.v` 完全相同的 sigma-delta 更新和初始相位 `(k*73)&255`。
- `lif_run()`：`V = V - (V>>leak) + bias + ΣW·spike`，`V>=θ` 时软复位。
- 硬件 TB 将 8 个测试样本的 `count0/count1/decision` 与
  `weights/hw_expected.mem` 逐项比较。

最终 xsim 日志：

```
CASE 0 PASS: decision=0 count0=15 count1=2 cycles=279714
CASE 1 PASS: decision=0 count0=14 count1=3 cycles=279714
CASE 2 PASS: decision=0 count0=14 count1=3 cycles=279714
CASE 3 PASS: decision=0 count0=13 count1=4 cycles=279714
CASE 4 PASS: decision=1 count0=2 count1=14 cycles=279714
CASE 5 PASS: decision=1 count0=3 count1=14 cycles=279714
CASE 6 PASS: decision=1 count0=3 count1=13 cycles=279714
CASE 7 PASS: decision=1 count0=3 count1=13 cycles=279714
=== RESULT: PASS (8/8 cases) ===
```

这 8 个用例覆盖 SNR -6.6..+2.7 dB、两类、不同置信度；硬件计数与 Python 完全一致，
说明权重打包、地址映射、编码器、LIF 更新、计数/判决全部位精确。
完整日志：`build/sim/sim_final.log`（8 个用例约 22.4 ms 仿真时间，即 279,714 cycles/例）。

---

## 6. 性能与资源

### 6.1 延迟/吞吐

- 每层每行：1 (LOAD) + 15 (chunk 1..15) + 1 (S_UPDATE) = 17 cycles。
- 每时间步：L1 512×17 + L2 512×17 + L3 2×17 = 17,442 cycles；
  加编码/握手约 17.4k cycles。
- 16 时间步 + 清空扫描共 **279,714 cycles**（实测 TB busy 计数，
  平均约 17.48k cycles/时间步，含清空开销）。
- @50 MHz：**5.594 ms/次，178.8 次/s**。
- 突触吞吐：525,312×16 / 5.594 ms ≈ **1.50 G 突触操作/s**；峰值 32×50 MHz = 1.6 G/s。
- 注意：5.6 ms 不是 100 µs PRI 的硬实时；适合低重频/离线演示，或后续做跨脉冲流水。

### 6.2 资源（Vivado 2025.2，`xc7z020clg400-2`，OOC）

| 资源 | 使用 | 可用 | 占比 |
|---|---:|---:|---:|
| LUT | 12,735 | 53,200 | 23.94% |
| FF | 12,650 | 106,400 | 11.89% |
| BRAM Tile | 128 | 140 | 91.43% |
| DSP | 0 | 220 | 0% |

- 权重 BRAM：L1 64×RAMB36，L2 63×RAMB36；膜电位 2×RAMB18；
  L3 权重/偏置走 LUTRAM。
- 编码器是 LUT 大户：约 9,244 LUT / 9,216 FF（512×9 个 sigma-delta 累加器）。

### 6.3 时序/功耗

- 约束：50 MHz（20 ns）。
- WNS **+2.853 ns**，WHS +0.106 ns，TNS/THS = 0。
- 最差 setup 路径约 17.03 ns，主要是编码器 tick 复制寄存器的控制分发；
  数据通路约 15.8 ns。100 MHz 需要进一步流水（或降 lane 数）。
- 功耗：vectorless 估算 0.375 W；没有真实翻转率，不能当板上实测值。

---

## 7. 复现

环境：

- Python：`D:\app\anaconda\envs\pytorch`（Python 3.10.15，torch 2.4.1+cu，RTX 4070 SUPER）。
- Vivado：`G:\app\vivado2025.2\2025.2\Vivado\bin\vivado.bat`（2025.2）。

```bash
# 1) 数据
python python/radar_data.py --bins 512 --train 24000 --val 4000 --test 8000 \
       --seed 2027 --out data/radar512.npz

# 2) 训练 ANN
python python/train_ann.py --data data/radar512.npz --h1 512 --h2 512 \
       --epochs 120 --seed 43 --wd 1e-2 --dropout 0.2 --noise 0.03 \
       --label-smooth 0.05 --out models/ann512_B.pt

# 3) 导出整数 SNN 权重 + 硬件测试向量
python python/export_hw.py --model models/ann512_B.pt --data data/radar512.npz \
       --pct 99.0 --T 16 --scale 1.003921568627451 --outdir weights

# 4) 仿真（必须在 build/sim 下运行，$readmemh 用相对路径）
cd build/sim
cp ../../weights/*.mem .
xvlog -sv ../../rtl/weight_mem.v ../../rtl/rate_encoder.v \
      ../../rtl/lif_layer.v ../../rtl/snn_top.v ../../sim/tb_snn_top.v
xelab -debug typical tb_snn_top -s tb_snn_top
xsim tb_snn_top -runall

# 5) 综合 + 实现（50 MHz，OOC）
cd ../synth
cp ../../weights/*.mem .
vivado -mode batch -nolog -nojournal -source ../../tcl/synth.tcl
vivado -mode batch -nolog -nojournal -source ../../tcl/impl.tcl
```

### 8-bit 效率版（推荐）

```bash
python python/train_ann.py --data data/radar512.npz --h1 256 --h2 128 --epochs 120 --seed 43 --wd 1e-3 --dropout 0.1 --noise 0.02 --label-smooth 0.05 --bias-init 0.1 --out models/ann_256_128.pt
python python/export_hw.py --model models/ann_256_128.pt --data data/radar512.npz --pct 99.0 --T 16 --scale 1.003921568627451 --w-bits 8 --lanes 32 --outdir weights_eff
cd build/sim_eff
cp ../../weights_eff/*.mem .
xvlog -sv ../../rtl/weight_mem.v ../../rtl/rate_encoder.v ../../rtl/lif_layer.v ../../rtl/snn_top_eff.v ../../sim/tb_snn_top_eff.v
xelab -debug typical tb_snn_top -s tb_snn_top_eff
xsim tb_snn_top_eff -runall
cd ../synth_eff
cp ../../weights_eff/*.mem .
vivado -mode batch -nolog -nojournal -source ../../tcl/synth_eff.tcl
vivado -mode batch -nolog -nojournal -source ../../tcl/impl_eff.tcl
```

### 4-bit 榨干版

```bash
# 训练（正偏置初始化，避免 704 宽 ReLU 层大面积死亡）
python python/train_ann.py --data data/radar512.npz --h1 704 --h2 704 --epochs 100 --seed 46 --wd 1e-4 --dropout 0.05 --noise 0.02 --label-smooth 0.03 --bias-init 1.0 --out models/ann704_c.pt

# 导出 4-bit / 64-lane 权重与测试向量
python python/export_hw.py --model models/ann704_c.pt --data data/radar512.npz --pct 99.0 --T 16 --scale 1.003921568627451 --w-bits 4 --lanes 64 --outdir weights4

# 仿真（注意 tb 文件叫 tb_snn_top4.v，模块名仍是 tb_snn_top）
cd build/sim4
cp ../../weights4/*.mem .
xvlog -sv ../../rtl/weight_mem.v ../../rtl/rate_encoder.v ../../rtl/lif_layer4.v ../../rtl/snn_top4.v ../../sim/tb_snn_top4.v
xelab -debug typical tb_snn_top -s tb_snn_top4
xsim tb_snn_top4 -runall

# 综合 + 实现
cd ../synth4
cp ../../weights4/*.mem .
vivado -mode batch -nolog -nojournal -source ../../tcl/synth4.tcl
vivado -mode batch -nolog -nojournal -source ../../tcl/impl4.tcl
```

报告位置：

> `build/` 是本地 Vivado 综合/仿真产物（约 57MB），未随仓库上传；按第 7 节命令可重新生成。
> `papers/` 只保留论文元数据/摘要 JSON；IEEE 全文闭源，EMRFormer 全文 PDF 也不随 public 仓库分发。

- `build/synth/util_impl.rpt`、`timing_impl.rpt`、`power_impl.rpt`
- `build/synth/post_route.dcp`
- `build/sim/sim_final.log`（8-bit 8/8 PASS 日志）
- `build/sim_eff/sim_eff.log`（8-bit 效率版 8/8 PASS 日志）
- `build/sim4/sim4.log`（4-bit 容量版 8/8 PASS 日志）
- `docs/metrics.json`（8-bit 精度/混淆矩阵/AUC/延迟）
- `docs/metrics_eff.json`（8-bit 效率版精度/混淆矩阵/AUC/延迟）
- `docs/metrics_cnn_mf.json`（MF+1D CNN 精度/混淆矩阵/AUC）
- `docs/metrics_hybrid_snn.json`（CNN 特征 + SNN 头精度/混淆矩阵/AUC）
- `docs/metrics4.json`（4-bit 容量版精度/混淆矩阵/AUC/延迟）
- `docs/metrics_spiking_cnn2d.json`（2D 时频 SNN 精度/PFA/脉冲率/延迟估算）
- `docs/metrics_cnn2d.json`（2D 时频 ANN 精度/参数量）
- `build/synth_eff/util_eff_impl.rpt` / `timing_eff_impl.rpt`（8-bit 效率版资源/时序）
- `build/synth4/util4_impl.rpt` / `timing4_impl.rpt`（4-bit 容量版资源/时序）
- `weights/meta.json`（最终网络、阈值、用例）

---

## 8. 目录

```
rtl/        snn_top.v / lif_layer.v / rate_encoder.v / weight_mem.v（8-bit 基线）
             snn_top_eff.v（8-bit 256-128 效率版）
             snn_top4.v / lif_layer4.v（4-bit / 64-lane 容量实验）
             conv2d_engine.v / rate_encoder2d.v / snn2d_top.v（2D 时频 RSN）
sim/        tb_snn_top.v / tb_snn_top_eff.v / tb_snn_top4.v / tb_snn2d_top.v
python/     radar_data.py / train_ann.py / convert_snn.py / eval_snn.py / export_hw.py
weights/    8-bit 基线权重、偏置、硬件测试向量、meta.json
weights_eff/ 8-bit 效率版权重、偏置、硬件测试向量、meta.json
weights4/   4-bit 容量版权重、偏置、硬件测试向量、meta.json
weights2d/  2D RTL 权重/偏置/阈值/FC/meta2d.json
hw2d/       2D RTL 8 个测试向量 + 期望分数/计数
tcl/        synth.tcl / impl.tcl / synth_eff.tcl / impl_eff.tcl / synth4.tcl / impl4.tcl
            synth2d.tcl / impl2d.tcl / synth2d50.tcl / impl2d50.tcl
build/sim/  8-bit 基线 xsim
build/sim_eff/ 8-bit 效率版 xsim
build/sim4/ 4-bit 容量版 xsim
build/sim2d/ 2D RTL xsim
build/synth/  8-bit 基线综合/实现
build/synth_eff/ 8-bit 效率版综合/实现
build/synth4/ 4-bit 容量版综合/实现
build/synth2d/ 2D RTL 100MHz 综合/实现
build/synth2d50/ 2D RTL 50MHz 综合/实现
docs/       metrics.json / metrics_eff.json / metrics4.json /
            metrics_cnn_mf.json / metrics_hybrid_snn.json / metrics_spiking_cnn2d.json /
            sim2d_8.log / report.html / report.pdf
models/     ann512_B.pt（基线）/ ann_256_128.pt（效率版）/ ann704_c.pt（4-bit）
            cnn_mf_max_r0.pt（MF+CNN）/ ann_hybrid.pt（CNN 特征 + SNN 头）
            cnn2d_tf.pt（2D 时频 ANN）/ spiking_cnn2d_tf.pt（2D 时频 SNN）
python/     ... + train_cnn.py / convert_cnn_snn.py / extract_cnn_feat.py /
            tf_data.py / train_cnn2d.py / train_spiking_cnn2d.py / eval_spiking_cnn2d.py
data/       radar512.npz / mf_cnn_feat.npz / radar_tf.npz（简单集）/ radar_tf_hard.npz（难集）
archive/    mf_mlp_experiment/radar_mf.npz（MF range profile 数据）
```

---

## 9. 下一步（纯 PL 上板 / 榨干）

当前 RTL 的接口是“512 字节特征写端口 + start/busy/done + 计数/判决输出”，
仿真和综合已完成。PS/DDR 等 PS 端资源不在本次范围，上板只做 PL：

1. **特征源**：片上 512 点 FFT + log + median/MAD 归一化，或外部简单接口预装特征。
2. **约束/引脚**：50 MHz PL 时钟、复位；生成 bitstream，上板实测延迟/功耗。
3. **实测**：真实雷达/回波数据、功耗、误报率；必要时重新校准阈值。
4. **性能/容量**：8-bit 256-128 效率版已完成（推荐，见 1.2）；混合方案（1.3）已试验。
   **时频 RSN（1.5）算法已验证 99.72%，128 lane/100 MHz 2D 引擎已完成（5.01 ms，
   见 1.6）**。下一步是跨 t 行缓冲 bit-exact、事件驱动（估算 0.79 ms）和上板。
   如果暂时不做 2D 引擎：保留 8-bit 256-128，或做 QCFS/PASCAL 1D spiking CNN。
5. **结果读出**：简单 PL 寄存器 / UART / ILA（可选）。
6. **时频前端**：片上 STFT（64 点、hop 16）+ log + median/MAD；或 PS 预处理后写 1,024 字节 TFG。
