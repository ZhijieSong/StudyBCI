# 数据清单 · BCI Competition IV · Dataset 2a（Graz data set A）

> 本清单是第 6/7 节「可运行管线 + 实测对比」所使用的原始数据说明。
> 我们仅用 **A01T.gdf**（subject A01 的训练 session，带标签）完成全部实测。

## 1. 数据集基本信息

| 项 | 值 |
|---|---|
| 名称 | BCI Competition IV · Dataset 2a（Graz data set A） |
| 官网 | <http://www.bbci.de/competition/iv/> |
| 说明（PDF） | <https://www.bbci.de/competition/iv/desc_2a.pdf> |
| 下载（GDF 打包，420 MB） | <https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip> |
| MNE 读取 | `mne.io.read_raw_gdf(..., preload=True)` |
| 作者 | C. Brunner, R. Leeb, G. R. Müller-Putz, A. Schlögl, G. Pfurtscheller（Graz） |

## 2. 下载文件清单

| 文件 | 含义 |
|---|---|
| `A01T.gdf` … `A09T.gdf` | 9 个被试的 **训练** session（**带类别标签**，可用于做对比/评估） |
| `A01E.gdf` … `A09E.gdf` | 9 个被试的 **评估** session（比赛时不公开标签，A01E 作为示例公开） |

本次实测只用 `A01T.gdf`（33,640,300 字节，GDF 1.99）。zip 压缩包约 420 MB（含全部 18 个文件）。

## 3. 记录参数（A01T）

| 项 | 值 |
|---|---|
| 采样率 | 250 Hz |
| 通道 | 25 = **22 EEG + 3 EOG**（前 22 为 EEG，最后 3 为 EOG） |
| EEG 电极 | 22 个 Ag/AgCl，电极间距 3.5 cm，国际 10-20 系统 |
| 参考 | **左乳突**（monopolar）；接地：右乳突 |
| 记录带通 | 0.5–100 Hz |
| 陷波 | **记录时已启用 50 Hz 陷波**（放大器层面） |
| EEG 灵敏度 | 100 µV |
| EOG | 3 个单极 EOG，250 Hz，带通 0.5–100 Hz + 50 Hz 陷波，灵敏度 1 mV；**仅用于伪迹处理，禁止用于分类** |

### 22 个 EEG 通道顺序（canonical montage）
```
Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6,
CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
```
> GDF 文件里只有部分通道带名字（Fz/C3/Cz/C4/Pz），其余为 `EEG`；
> 我们用上面这份顺序对前 22 个通道重命名（已验证带名通道位置 Fz@0, C3@7, Cz@9, C4@11, Pz@19 完全吻合）。

## 4. 实验范式（cue-based 4 类运动想象）

| 时间 | 事件 |
|---|---|
| t = 0 s | 屏幕出现注视十字 + 短促提示音 → **trial 开始**（事件 768） |
| t = 2 s | 出现箭头（左/右/下/上）→ **cue**（事件 769/770/771/772 = 类别 1/2/3/4），箭头停留 1.25 s |
| t = 2–6 s | 被试执行对应运动想象 |
| t = 6 s | 注视十字消失，短休 |

**四类**：1 = 左手想象，2 = 右手想象，3 = 双脚想象，4 = 舌头想象。

每 session：6 个 run × 每 run 48 个 trial（每类 12 个）= **288 trials**；
每 run 间隔 100 个缺失值（MNE 已处理）。A01T 里 288 个 cue、每类 72 个。

## 5. 事件码（GDF，desc_2a.pdf Table 2）

| 十进制 | 十六进制 | 含义 |
|---|---|---|
| 768 | 0x0300 | trial 开始 |
| 769 | 0x0301 | cue 左手（class 1） |
| 770 | 0x0302 | cue 右手（class 2） |
| 771 | 0x0303 | cue 双脚（class 3） |
| 772 | 0x0304 | cue 舌头（class 4） |
| 1023 | 0x03FF | 专家标记的伪迹 trial（应剔除） |
| 276 / 277 | 0x0114 / 0x0115 | 睁眼 / 闭眼（EOG 标定） |
| 1072 | 0x0430 | 眼动（EOG 标定） |
| 32766 | 0x7FFE | 一个新 run 开始 |

**A01T 统计**：288 个 trial-start（768）、288 个 cue（769–772，每类 72）、15 个专家标记伪迹（1023）、9 个 run 开始（32766）。
> 本次实测统一剔除 15 个专家标记伪迹 trial（对应 288−15 = **273 个干净 trial**），以保证各配置可比。

## 6. 分类评估口径（本次实测）

- 4 类 **CSP + LDA**，One-vs-Rest CSP（4 类 × 4 个空间滤镜 = 16 个 log-方差特征），LDA 分类。
- 分层 **5 折交叉验证**（随机种子固定，保证可复现）。
- 只使用 22 个 EEG 通道；EOG 不进入分类。
- 特征时间窗：cue 后 **0.5–4.0 s**；基线校正窗 −1.5–0 s。
- 报告 **准确率（均值±标准差）** 与 **CSP 判别信噪比（dB）**（= 每类 one-vs-rest 广义特征值最大比值平均后取 dB）。

## 7. 复现（命令行一次性跑整份对比）

```bash
pip install -r requirements.txt
python scripts/run_grid.py          # 重跑第 7 节对比表 -> results/comparison_table.csv
python scripts/make_figures.py      # 生成频谱/蒙太奇/结果图 -> results/*.png
```

在 notebook 里：`preprocessing_pipeline.ipynb` → Run All（`A01T.gdf` 与本文件同目录，或改 `DATA` 为绝对路径）。
