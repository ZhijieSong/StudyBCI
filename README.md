# StudyBCI · 把 EEG 预处理做对

配套公众号文章《把 EEG 预处理做对》的可复现代码与实测结果。给出一套**能直接跑的 MNE 预处理管线**，以及在同一份数据（BCI Competition IV 2a · A01T）上「带通 ± / 陷波 ± / ICA ± / 重参考方式」的**真实准确率与信噪比差异**。所有数字均为实测、可一键复现，不是文献摘抄。

> 面向：已经会读一点脑电波形、想在 BCI 上做出可复现结果的工程师与研究者；也适合入门研究生对照学习。

---

## 目录结构

```
StudyBCI/
├── notebooks/
│   └── preprocessing_pipeline.ipynb    # 主入口：原始 GDF → 干净 epoch + 第 7 节对比表 + 频谱图（Run All）
├── src/eeg_preproc/
│   ├── __init__.py
│   └── preprocess.py                    # 预处理管线核心模块（重参考/陷波/带通/ICA/分段/分类/SNR）
├── scripts/
│   ├── run_grid.py                      # 命令行一键重跑第 7 节对比表
│   └── make_figures.py                  # 生成频谱 / 蒙太奇 / 结果图
├── docs/
│   ├── data.md                          # 数据清单：官方链接、记录参数、范式时序、事件码、通道序
│   └── parameter-guide.md               # 参数模板：每个预处理的推荐值 + 取舍说明
├── results/
│   ├── comparison_table.csv             # 第 7 节实测对比表（结构化数据）
│   ├── noise_analysis.txt               # 50 Hz 线噪 / 各频带功率数值
│   └── figures/                         # 频谱对比图、蒙太奇图、准确率 + SNR 结果图
├── data/                                # 原始数据（不随仓库分发，见 data/README.md）
├── requirements.txt
├── LICENSE
└── README.md
```

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 下载数据：BCI IV 2a 官方公开数据集（见 docs/data.md），把 A01T.gdf 放到 data/
#    （示例：data/A01T.gdf）

# 3a. 方式 A：打开 notebook 运行
jupyter notebook notebooks/preprocessing_pipeline.ipynb   # Run All

# 3b. 方式 B：命令行复现
python scripts/run_grid.py       # -> results/comparison_table.csv
python scripts/make_figures.py   # -> results/figures/*.png
```

换到自己的数据：改 `data/A01T.gdf` 为你的文件，并按 `docs/parameter-guide.md` 逐项调整（参考方式、带通频段、ICA 顺序、通道/事件映射）。

## 核心结论（同一份数据 · 第 7 节实测）

口径：A01T · 273 个无伪迹 trial · 4 类 **CSP + LDA** · 分层 **5 折交叉验证** · 特征窗 0.5–4 s。

| 配置 | 参考 | 带通 | 陷波 | ICA | 准确率 | 信噪比(dB) |
|---|---|---|---|---|---|---|
| C1 几乎不预处理 | mastoid | 宽带(1-100) | 无 | 无 | **62.7%** | 1.12 |
| C2 CAR 宽带 | CAR | 宽带 | 无 | 无 | 62.7% | 1.30 |
| C3 CAR 宽带 + 陷波 | CAR | 宽带 | 50Hz | 无 | 62.7% | 1.30 |
| C4 CAR 宽带 + 陷波 + ICA | CAR | 宽带 | 50Hz | 是 | 64.1% | 1.29 |
| **C5 推荐管线** | **CAR** | **8-30Hz** | 无 | 是 | **73.6%** | **1.91** |
| C6 CAR 8-30（无 ICA） | CAR | 8-30Hz | 无 | 无 | **76.9%** | 1.90 |
| C7 mastoid 8-30 | mastoid | 8-30Hz | 无 | 是 | 69.9% | 1.76 |
| C8 laplacian 8-30 | laplacian | 8-30Hz | 无 | 是 | 72.5% | 1.74 |

- **带通是最重要的杠杆**：宽带(1–100 Hz) → 8–30 Hz，准确率 **62.7% → 76.9%（+14.2 点）**，SNR +0.60 dB。
- **陷波在这份数据上几乎无效**：放大器记录时已做 50 Hz 陷波（实测频谱 50 Hz 峰/邻比 = 0.367，是凹口而非线噪峰）。
- **重参考**：CAR(73.6%) ≈ 拉普拉斯(72.5%) > 乳突(69.9%)。
- **ICA（去 EOG）影响较小且双向**：宽带 +1.4 点、窄带 −3.3 点（±1σ 内）——**别把任务脑电当伪迹剔掉**。

> 注意：这是 **A01T 单受试者、跨 trial 的 5 折 CV**，不是 9 人平均。不同受试者 / 数据集会不同，请以你自己数据的实测为准。

## 在线性能 / GPU 加速（文章 §10–12 数字来源）

特定硬件实测（CPU i5-12490F | GPU RTX 3060 8GB），用于判断「在线能不能跑、要不要上 GPU」：
- 在线延迟主要由**数据积累窗 + 因果滤波群延迟**主导，不是算力（因果 IIR 2 阶群延迟 ≈ 18–41 ms；离线零相位 `filtfilt` 群延迟为 0 但**在线不可用**）。
- GPU 要在单次运算大到**盖过 CPU↔GPU 搬运**（22 导一趟约 30–60 ms）才值得；预处理/特征 CPU 已够快。
- GPU 主战场是深度学习：EEGNet 训练 / 推理 **13.5×**。
- 降采样务必用带抗混叠滤波的重采样（裸 `data[::k]` 会把高频折返进带内，实测能量高 +141 dB）。

## 版权与数据

- 数据：BCI Competition IV Dataset 2a（Graz 工业大学公开数据集，官方链接见 `docs/data.md`）。
- 代码与结果：配套《把 EEG 预处理做对》交付，详见 `LICENSE`。
