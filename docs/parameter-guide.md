# 参数模板 · MNE 预处理管线（BCI IV 2a 运动想象）

> 直接抄去改就能用。每行给出「参数 / 推荐值 / 说明与取舍」。
> 对应代码：`scripts/preprocess_pipeline.py` 与 `preprocessing_pipeline.ipynb`。

## 1. 数据与通道

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `gdf_path` | `'A01T.gdf'` | BCI IV 2a 训练/评估文件，或你自己的 GDF/EDF |
| `CANON`（22 通道序） | `Fz,FC3,FC1,FCz,FC2,FC4,C5,C3,C1,Cz,C2,C4,C6,CP3,CP1,CPz,CP2,CP4,P1,Pz,P2,POz` | 按此顺序把前 22 通道重命名（GDF 里多数通道无名字） |
| montage | `standard_1020` | 10-20 国际系统；用于 Laplacian/双极与蒙太奇图 |
| 通道类型 | 22 EEG 设 `eeg`，3 EOG 设 `eog` | EOG 只用于伪迹，不进分类 |

## 2. 重参考（第 2 节 · 怎么选）

| 方法 | 参数 | 说明 |
|---|---|---|
| 不重参考（保留左乳突原参考） | `reference='mastoid'` | 记录时即乳突参考；共性噪声大 |
| **CAR（公共平均参考）⭐推荐** | `reference='CAR'` | `set_eeg_reference('average')`；去掉全部通道共有成分，对 MI 通常最稳 |
| 表面拉普拉斯 / CSD | `reference='laplacian'` | `compute_current_source_density`；近似参考无关、突出局部（本数据实测亦可，见第 7 节） |
| 双极 | （未在本实验单独测量） | 用相邻通道相减，如 `set_bipolar_reference(anode, cathode)`；对特定导联对有效 |

## 3. 滤波（第 3 节 · 频段 / 陷波 / 顺序）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| 高通（去漂移 / ICA 前置） | `highpass=1.0` Hz | 先于 ICA；滤掉慢漂移，也让 ICA 更稳 |
| 带通（任务频段） | `bandpass=(8, 30)` | MI 常用 μ(8–13)+β(13–30)；**不是越宽越好**（第 7 节：宽带 1–100Hz 比 8–30Hz 差约 14 个点） |
| 备选带通 | `(4, 38)` / `(8, 40)` / `(4, 10)` 等 | 视频段/范式定；SSVEP 用更窄的刺激频段，P300 用更低频 |
| 陷波 | `notch=None` | **本数据记录时已陷 50Hz**，再切无益（第 7 节实测：无差异）。若确有线噪用 `notch=50` |
| 陷波宽度 | `notch_widths=1.0` Hz | 切太宽会伤 α；太窄切不净 |
| 滤波器设计 | `fir_design='firwin'`（零相位 FIR） | 默认即可；IIR 省时但相位非线性 |
| 顺序 | 重参考 → 陷波 → 高通 → **ICA** → 任务带通 → 分段 | **ICA 放在窄带通之前**，否则眨眼/眼动低频分量已被滤掉，ICA 找不到（第 7 节实测印证） |

## 4. 伪迹剔除（第 4 节 · ICA）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `ica` | `True` / `False` | 是否做 ICA 去 EOG 伪迹 |
| `n_components` | `None`（=通道数，PCA 后再 ICA） | 也可手动指定成分数 |
| `method` | `'fastica'` | 默认 |
| `ica_threshold` | `2.5` | `find_bads_eog` 的 z-score 阈值；越大越保守 |
| EOG 参考通道 | `['EOG-left','EOG-central','EOG-right']` | 用 3 个 EOG 通道定位眼动成分 |
| **取舍** | — | 第 7 节实测：本数据只剔 2–3 个 EOG 成分，对准确率影响**很小甚至略负**；务必检查被剔成分是否带了任务相关信号——**别把运动想象成分当伪迹剔掉** |

## 5. 分段与基线（第 5 节）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `tmin` / `tmax` | `-1.5` / `4.5` s | 相对 cue；覆盖 cue 前基线 + 完整想象期 |
| `baseline` | `(-1.5, 0)` s | cue 前注视期做基线校正 |
| 特征窗（分类用） | `(0.5, 4.0)` s | 避开 cue 诱发响应；如需更贴近在线可前移 |
| 剔除 | 专家标记伪迹 trial（事件 1023） | 与比赛「只评干净 trial」口径一致，保证各配置可比 |
| 幅值拒绝 `reject` | `None` | 本实验统一不去幅值，保证各配置 trial 数一致 |

## 6. 分类器（第 7 节评估口径）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| 特征 | One-vs-Rest CSP 的 **log-方差** | 4 类 × `n_per_class=4` 个滤镜 = 16 维 |
| `n_per_class` | `4` | 每类保留的空间滤镜数；越大特征越多、越易过拟合 |
| CSP 正则 `alpha` | `0.01` | 协方差收缩；**必须**，因 CAR 之后协方差秩亏（rank 21/22） |
| 分类器 | `LinearDiscriminantAnalysis` | 经典、省样本 |
| 交叉验证 | 分层 5 折（种子固定） | 报告均值±标准差 |

## 7. 实测口径（第 7 节 · 复现）

- 数据：A01T，22 EEG 通道，剔除 15 个专家伪迹 trial → **273 trial**。
- 基线（推荐）管线：`CAR` + `bandpass=(8,30)` + `ica=True` + 高通 1 Hz。
- 对照表见 `results/comparison_table.csv`；一键重跑：`python scripts/run_grid.py`。

## 8. 在线性能 / GPU 实测口径（§10 / §11 / §12 · W-131 · 本数据实测）

> 复现脚本与完整数字在实验员另一归档 `E:\CherryClaw\projects\bci-content-lab\在线性能与GPU加速实测\`：`scripts/measure_*.py`、`results/*.json`、`results/README_在线性能_GPU实测.md`。
> 硬件：CPU i5-12490F（6 核/12 线程）｜GPU RTX 3060 8GB（torch 2.6.0 + cu124 / CUDA 12.4）。数据 A01T / 22 导 / 250 Hz / 273 clean MI trial。

| 维度 | 参数 / 数值（本数据实测） |
|---|---|
| 在线滤波群延迟 | 因果 Butterworth 带通 8–30 Hz @250 Hz：2 阶 10/15.5/20 Hz ≈ 41/20/18 ms；4 阶 ≈ 81/37/32 ms。A01T Cz 主频 12.2 Hz 包络时移：2 阶 20 ms、4 阶 40 ms。`filtfilt` 零相位群延迟 = 0（非因果，仅离线） |
| 延迟预算 / RTF | CAR → 因果 IIR → CSP 投影 → log方差 → LDA，决策窗 3.5 s = 875 采样；每块流式 0.035–0.087 ms、单次决策 2.12 ms，RTF ≈ 0.0002–0.0006，预处理占单次决策约 2–4% |
| CSP / ICA 在线投影 | CSP 决策窗 68.5 µs / 单块 1.7 µs；ICA 决策窗 69.1 µs / 单块 2.2 µs；273 epoch 批量 CSP 52.3 ms / ICA 74.9 ms |
| CPU vs GPU | CPU=numpy/scipy float64，GPU=torch float32。CAR 42.2 vs 59.8 ms（含搬运）、FIR 712 vs 78 ms、CSP 804 vs 35.3 ms、CCA/TRCA 776 vs 28.1 ms、频带功率 32 vs 12.5 ms（GPU 纯算依次 0.61/20.1/25.2/12.3/0.19 ms，见 README） |
| EEGNet 训练 / 推理 | 4 类、273 epoch、模型 4412 参 / 17.2 KB：训练每 epoch CPU 0.995 s vs GPU 0.074 s（13.5×）；推理 batch=64 CPU 55.3 ms vs GPU 4.11 ms（13.5×） |
| 降采样抗混叠 | 250 → 100/128 Hz：27.5–48 Hz 带内裸 `data[::k]` 比 MNE `resample` 能量 +1.6 / +2.3 dB；注入 70 Hz → 折返 30 Hz，裸抽取比 MNE +141 dB |

**结论一句话**：在线延迟被数据积累窗 + 因果群延迟主导（不是算力）；GPU 单次运算要大到盖过 CPU↔GPU 搬运（本机 22 导一趟 ~30–60 ms）才值得上；EEGNet 训练 / 推理是 GPU 主战场（13.5×）。
