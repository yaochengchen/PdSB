# PdSB：Potts 离散模拟分岔（2× GH200）

**PdSB**（Potts discrete Simulated Bifurcation）：把 dSB 的自旋换成正单纯形上的向量。
每种颜色 c 对应 R^(K−1) 中正单纯形的一个顶点 e_c，任意两个不同颜色的内积都是 −1/(K−1)，所以对任意 K，
Potts 能量都是精确的（不像 clock/planar 那样只在 K≤3 时精确）。dSB 的盒子 [−1,1] 换成单纯形凸包上的非弹性墙：
凸包上 |x| 最大的点正好是各个顶点，pump 项会把状态推到离散态上，不需要相位离散化参数，也不需要 one-hot 惩罚。
K=2 时就是普通的 dSB。连续耦合的变体叫 **PbSB**（对应 Goto 的 bSB）。

论文定位：**提出 PdSB 这个新算法**。Frederiksen, De Prins, Bienstman, arXiv:2605.06425 (2026) 测了 5 种 Potts 机，
全部输给 one-hot Ising 机（q=3 到 q=4 差距还变大），并把原因归到 planar Potts 的角度约束上；PdSB 去掉了这个约束，
这个结论作为动机，并在他们的协议下正面回应。他们的参考 IM 主要用**发表的数值**（Supp. Tables S7.28/29）比较，
只在发表数据回答不了的地方自己跑（复现检查、时间–质量）。他们自己也写明没有研究 time-to-solution。

## 方法（同一个 PyTorch 框架、同一块 GPU、同一种 GEMM 精度）

| 名字 | 是什么 | 作用 |
|---|---|---|
| `pdsb_fused` | PdSB，融合 CUDA kernel（结构同 cdsb：GEMM → 融合 kernel → reduce，CUDA graph） | 我们的最终实现 |
| `pdsb` | PdSB，纯 PyTorch | 和 ref-IM 做同框架公平比较 |
| `pbsb` | PbSB（连续耦合） | 变体 |
| `ref_im` | 2605.06425 的参考 Ising 机，Eq. (9) 原样实现：sigmoid CIM + spin-sign，one-hot（Eq. S1.33），外场乘 ζ，Euler–Maruyama，α=−50，β=ε_β t，dt=1e−3，f_ξ=1e−4，**1e6 步**，**他们的超参网格**（Tables S6.25/26） | 只做复现检查（G1，K=3/4）和 E2 计时；主表用他们发表的数值 |
| `onehot_dsb` | 同一个 dSB 积分器，只把编码换成 one-hot | 消融：编码 |
| `pdsb_sphere` | 单纯形嵌入 + 球面墙 | 消融：墙 |
| `clock_dsb` | clock 嵌入 + 圆形墙（planar Potts，以前所有 Potts 机都属于这一类） | 消融：嵌入；复现他们看到的 K=4 崩溃 |
| `sa` | CPU heat-bath Potts SA（numba，Grace 核） | 经典参照 |

- 整个超参网格放在**一个 batch** 里跑：每个 replica 有自己的参数，所以一个 job = (方法, 图, K, 步数)。
- 所有离散状态之间的耦合乘积都是整数（one-hot 颜色、自旋符号、±1 权重），因此用 FP16 tensor core 做 GEMM 也是精确的（检查条件：整数权重、行绝对值和 ≤ 2048）。ref-IM 同样享受这一点。报告的割值一律重新精确计算；融合 kernel 在设备上逐步跟踪的最好值也会和精确值核对（`tracked_mismatch` 必须为 0）。
- 指标和 2605.06425 一致：每次运行沿轨迹看到的**最好割**，相对 BKS 的间隙（mean / median / min / SD），每个设定 100 次运行，按平均间隙为每张图选超参。

### ref-IM 复现中需要说明的假设（论文没写明的地方）

- 编码里 A=1、B=(B/A)n/n（论文只给出 (B/A)·n 的扫描范围）；x(0)=0（原点）。
- 颜色取 argmax_c x_uc：永远是合法着色，这是对 IM 最有利的解码方式。
- "每次运行的最大割"每 10 步评估一次（在 CUDA graph 内完成）。
- E1 的复现检查表会把我们的 ref-IM 和他们发表的数值逐图对比；对得上，才说明比较是公平的。

## 实验计划（按重要性排序）

`python run.py --plan all` 分三轮：**第 1 轮** E1、E3、E4（固定网格）→ **第 2 轮** E1v、E3v（从第 1 轮选设定）→ **第 3 轮** E2L、E4L（计时）。

| 计划 | 内容 |
|---|---|
| `E1` | G1–G5、K=3,4，所有方法各自扫网格（2605.06425 的协议，每个设定 100 次）。2026-10-14 把最优落在网格边上的方法都加宽了（SA 的 T0/T1、one-hot 的 ζ/A、球面墙、clock、PbSB、PdSB）。ref-IM：G1 上他们的完整网格 + 往 (B/A)n 更大方向的扩展（我们的最优在他们网格的上边界）；其他图只保留已经跑完的 |
| `E1v` | E1 每个 (方法, 图, K, 步数) 的最好设定，新种子 1000 次：无偏的数字、Mann–Whitney 检验、TTS99；ref-IM 只在 G1 |
| `E2` | 旧的计时（每档预算 100 次，GPU 没吃满），只作参考，不在 `all` 里 |
| `E2L` | **大批次计时**：每档预算 1000 次，E1 选出的每图设定；fused PdSB、PyTorch PdSB、one-hot dSB、SA（G1–G5），ref-IM（G1）。CPU 任务等 GPU 任务全部结束后单独跑，SA 计时不受干扰 |
| `E3` | K=5、6、8（G1–G5），加宽后的网格；不跑 ref-IM |
| `E3v` | E3 里 PdSB fused / SA / one-hot dSB 在 T=1e5 的最好设定，新种子 1000 次（检验 + 存档着色） |
| `E4` | 大图 G43、G22、G55、G70、G81，K=3,4：PdSB 和 SA **各自按图调参**（以前 top-3 的设定在网格里，不重跑）。n > 4096 走稀疏 kernel |
| `E4L` | 大图计时：E4 每个 (方法, 图, K) 的最好设定，200 次；PdSB 预算 1e3–1e6（2026-10-15 加了 3e5、1e6：和 SA 1e5 同墙钟时间需要约 1e6 步 PdSB），SA 1e3–3e5（合法着色实例不跑 3e5）；CPU 在 GPU 之后单独跑。报告里另有"同墙钟时间"对比表 |

**2026-10-15 再次加宽网格**（上次结果里最优仍在边上的地方；按设定续跑，只跑新加的点）：
SA（E1、E3）T1 加 0.3–0.5，T0 加 0.5 和 6.0；SA（E4）往冷的方向加 T0 0.25–0.5 × T1 0.01–0.05，G22/G43 加 T1=0.3（合法着色实例 G55 K4、G70 不加）；
one-hot（E1、E3）在该 K 选中的 ζ（A=2）上扫 ξ 0.25–1.0 × dt 0.35–1.0；PdSB（E3）加 ξ 1.75、2.0；PdSB（E4）加 dt 1.25–2.0。

### PdSB 找最好解（单独的结果文件夹）

| 计划 | 内容 |
|---|---|
| `H1` | 22 个目标实例（G22 K3/4、G43 K3/4、G55 K3、G81 K3/4、G1–G5 K5/6/8），fused PdSB，T=1e6，25 个设定 × 40 次（长时间尺度上重新调参） |
| `H2` | 每个目标取 H1 里按平均割最好的 2 个设定 + 按最大割最好的 1 个，T=1e6，每个 1000 次 |
| `H3` | 每个目标取 H1 最好的 1 个设定，T=1e7，200 次 |

- `python run.py --plan hunt --results results_pdsb`（不写 `--results` 时 hunt 自动用 `results_pdsb`）。分两轮：H1 → H2+H3。
- 这个文件夹只跑 PdSB，所以 `results_pdsb/best/` 里存的全是 PdSB 找到的着色，可以直接回答"PdSB 自己能不能找到"。
- hunt 的任务会对轨迹上的最好着色再做一次 1-opt（`best_polished`），结果仍然是 PdSB(+1-opt)。
- `python hunt_report.py results_pdsb --main results`：每个目标列出文献值、主结果存档的最好值（以及谁找到的）、PdSB 在主结果里的最好值、hunt 的最好值，以及每个计划/步数下达到目标的次数。
- run.py 不允许把 hunt 和 E 计划写进同一个文件夹。**不要和主计划同时跑**（E2L/E4L 是计时实验，GPU 被占会污染时间）。
- 粗估（按 E4L/E3v 实测的每步耗时）：H1 约 0.5 h，H2+H3 约 2.5 h（两张卡）。

粗估（按已跑结果的实测速度）：第 1 轮约 1.3 h（CPU 上的 SA 是瓶颈，E4 的 G81 最慢），第 2 轮约 0.3 h，第 3 轮约 0.6 h，合计约 2.5 h。

**续跑按设定（setting）粒度**：计划里的每个 job 只跑同一个 (方法, 图, K, 步数, 种子, 次数) 下**还没跑过的设定**。
所以加宽网格只会跑新加的点；被打断（Ctrl+C / kill）的 job 没有写进 `runs.jsonl`，下次自动重跑。
分析时，E1v、E2L、E3v、E4L 只用**当前**选出的设定（网格加宽后最优可能变了，旧选择的行会被忽略，避免"多次新种子里挑最好"的偏差）。
`analyze.py` 读取时跳过被截断的行和重复记录的同一个 job。**直接 kill 主进程时，spawn 出来的 GPU worker 可能还活着**：重启前先跑 `python check_results.py`。

**稀疏路径**：fused PdSB 在 n ≤ 4096 时用稠密 FP16 GEMM（tensor core，精确），n > 4096 时用 CSR gather kernel
`H[i,r,c] = Σ_j w_ij [col_j,r = c]`（FP32，固定求和顺序，整数权重时精确），直接读颜色，不需要 one-hot 缓冲。
两条路径逐位相同（selftest 第 11 项）。阈值可以用环境变量 `PDSB_DENSE_MAX` 改。

**公平性**：SA 用两颗 Grace 的约 136 个线程，PdSB 只用一张 GPU（对 SA 有利）。各方法网格是否够宽，见报告附录的 grid-edge 表。

**超过 BKS 的解**：任何一次运行的精确割超过已知最好值，就把着色存到 `results/<plan>/certificates/`，日志里会标 `** BEATS BKS **`。
可以用 `python verify_cert.py results/*/certificates/*.npy` 独立复核：直接读 Gset 原始文件，用纯 Python 逐条边重算割值。

**最好解存档**：不管有没有 BKS（包括 E3 的 K=5/6/8 和 E4 的大图），每个 (图, K) 只要出现比存档更好的精确割值，
就把着色存到 `results/best/<图>_K<K>.npy`，来源（计划、方法、步数、设定、replica）写在同名 `.json` 里，每次刷新还会追加到 `results/best/history.jsonl`。
存档前会用 fp64 按边重算割值；GPU 和 CPU 的 worker 同时写时有文件锁。`python verify_best.py` 会直接读 Gset 原始文件逐边复核，并和 BKS 比较。

## 运行

```bash
conda activate matching-cd
bash check_env.sh                   # 只读检查；缺包时 bash check_env.sh --install 只装进这个环境
python selftest.py                  # 11 项；第 9–11 项需要 GPU（10: fused vs PyTorch；11: 稀疏 kernel vs 稠密 GEMM）
python bench.py --graph G1 --K 4    # 实测每步耗时，用来核对上面的时间估计
python run.py --plan smoke --results results_smoke   # 约 2 分钟，跑通整个流程（不碰正式结果）
python check_results.py             # 每次重启前：检查残留进程、runs.jsonl 是否有截断/重复行、各计划还剩多少 job
nohup python run.py --plan all > all.log 2>&1 &     # 三轮：E1 E3 E4 → E1v E3v → E2L E4L
python analyze.py results           # -> results/report.md、summary.csv、figures/
# PdSB 找最好解（在主计划跑完之后）
python run.py --plan hunt --results results_pdsb > hunt.log 2>&1
python hunt_report.py results_pdsb --main results
python verify_best.py --best results_pdsb/best
```

- 两张 GPU 各跑一个 worker（`--gpus 0 1`），并尽量绑定到各自 GPU 所在 NUMA 节点的 Grace 核；SA 同时在 CPU 上跑（`--cpu-threads`）。
- 中断后重跑只跑还没完成的设定（每个计划各有一个 `results/<plan>/runs.jsonl`）。
- 也可以单独跑：`E1` / `E3` / `E4` 随时可跑；`E1v`、`E2L` 需要 E1，`E3v` 需要 E3，`E4L` 需要 E4。

### 融合 kernel 的编译

第一次调用 `fused.ext()`（selftest 第 10 项）时，会用 `torch.utils.cpp_extension` 编译 `csrc/pdsb_fused.cu`，大约 1 分钟，结果缓存在 `~/.cache/torch_extensions`。
需要 `nvcc`，且它的 CUDA 主版本要和 `torch.version.cuda` 一致（`check_env.sh` 会打印两者）。nvcc 不在默认位置时，用 `export CUDA_HOME=/usr/local/cuda-12.x` 指定。
GH200 可以设 `PDSB_ARCH=9.0`，只编 sm_90，编译更快。
我在 CUDA 13.0 下对 sm_90 编译通过：K=3/4/5/6/8 各个实例都没有寄存器溢出（47–122 个寄存器）。
融合方案的逻辑（ping-pong 缓冲、由 GEMM 顺带算出每步能量、延迟一步提交 best）已经用 NumPy 仿真和 PyTorch PdSB 逐步核对过。
fused kernel 已在 GH200 上通过 selftest 第 10 项。2026-10-14 新加的稀疏 gather kernel 用 clang（CUDA 12.9 头文件，sm_90）编译通过、无寄存器溢出，但还没在 GPU 上跑过：以 selftest 第 11 项为准（稀疏和稠密两条路径必须逐位相同）。

## 文件

| 文件 | 作用 |
|---|---|
| `machines.py` | PdSB（含 PbSB / sphere / clock 变体）、OneHotSB、RefIM、DeviceGraph（精确代价、1-opt）、带 CUDA graph 的积分器 |
| `fused.py`、`csrc/pdsb_fused.cu` | 融合 CUDA 版 PdSB（torch 扩展，K=3,4,5,6,8） |
| `sa_potts.py` | numba heat-bath Potts SA |
| `maxkcut.py` | Gset 读取/下载、BKS（G1–G5）、2605.06425 发表的数值 |
| `plans.py` | E1–E4 的网格和 job 列表（ref-IM 缩减版，见上） |
| `run.py` | 双 GPU 调度、断点续跑、计划之间的参数选择 |
| `analyze.py` | 主表（他们的格式）、复现检查、E1v 显著性检验、E2 速度–质量曲线和 TTS99、E3、E4 |
| `selftest.py`、`bench.py`、`check_env.sh`、`verify_cert.py` | 自检、吞吐量测试、环境检查、超过 BKS 的解的独立复核 |
| `check_results.py` | 重启前检查（残留进程、截断/重复行、剩余 job、已保存的解）；`--clean` 去掉坏行（原文件备份） |
| `verify_best.py` | `results/best/` 最好解存档的独立复核 |
| `hunt_report.py` | PdSB 找最好解（H1–H3）的报告：和文献值、主结果存档比较 |
