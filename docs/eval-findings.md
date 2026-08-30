# 《晴天》前 8 小节匹配评测 — 实验结论（2026-08-30）

评测命令（基线配置 := `--attack-weight 0.35 --harmonic-ratio 0.58 --log-compress`）：

```
.venv/bin/python scripts/match_answer.py \
  --audio "/Users/youzi/Downloads/晴天吉他谱-指弹谱-g调-虫虫吉他.mp3" \
  --answer "/Users/youzi/Downloads/晴天_1-8小节_TAB_含击勾弦.txt" \
  --bars 8 --attack-weight 0.35 --harmonic-ratio 0.58 --log-compress
```

基线锚点：**strict=0.3857, midi=0.6714, seq=0.5571, offset=25.0s, notes=1714**（70 个答案事件）。

## 已证伪的路径（全部卡死在 ≤0.39 strict）

| 尝试 | 结果 |
|---|---|
| octave_ratio 八度单独放宽（0.05–0.58） | midi 完全不变 0.6714；目标旋律在到达 suppression 前已被过滤 |
| baseline_percentile 中位地板放宽（50→0） | midi +0.014，strict −0.16 |
| novelty_weight 起音新颖度加权 | 全档位更差 |
| local_max_radius 半音 NMS | 1 无效果，2 更差（已 revert） |
| re_onset_gate 持续音门禁 | 灾难性（0.086，已 revert） |
| freq_weight 频率补偿 f^γ | 更差 |
| bin_normalize 逐 bin 归一 | 更差（已移除） |
| STFT 前端替换 CQT | 0.2571/25.0，不如 CQT |
| 目标时间对齐 onset（±0.12/0.21s） | 30→30, 无效 |
| 每小节独立 offset（8 自由度过拟合） | 上限 exact=33/70 |
| basic-pitch（Spotify SOTA）默认参数 | exact 仅 15–18/70 |
| basic-pitch 极限放宽阈值 | 55/70 但 8384 个音全时间段都有 40+ 命中 → 稀释假象 |
| demucs 分离 |  vocals/bass stem ≈ 静音（0.0001）：**该 MP3 就是纯吉他**，无可分离；other 干声评测 midi 0.7143 但 strict 0.2714 |

## 已证实的事实

1. 旋律音确实存在于音频：E4=331Hz 在 27.05s 处是 STFT 最强峰（幅度 212），G4=390Hz 同样清晰。
2. 音频调性 = G 大调（chroma 相关 0.806），与答案 TAB 调性一致。
3. offset=25.0 是正确的结构对齐点（全曲扫描 + 每小节扫描均支持）。
4. 演奏是 16 分音符网格（onset IOI 中位 ≈0.21s@71.8bpm）。
5. CQT 前端系统性低估中高寄存器：27.05s 处 64 的 CQT 强度 2.27 vs 52 的 16.6（STFT 却显示 E4 最强）。constant-Q 长窗对低频累加天然不公平。
6. 检测输出里大量持续音重复发射 + 相邻半音垃圾音（41/44/69/71），抢占 max_polyphony=6 名额。
7. 答案 TAB 的小节内时间是 order/span 等分估计，16/70 与最近 onset 偏差 >0.15s。

## 结论

**strict recall ≥ 0.9 在"检测器侧的渐进调参"路线上不可达。** 两个独立检测器（自研 CQT 管线、basic-pitch）在该音频上的 exact-midi 上限都只有 ~30–33/70。剩余差距来自密集指弹混合音频的本质区分难度（同和弦多音 + 八度遮蔽 + 持续音复触发），以及答案 TAB 与演奏实况之间无法排除的细节差异。

## 若未来要继续，可行方向（按性价比）

1. **答案约束转录**（此曲专用）：以答案 TAB 为候选空间，用音频证据选时间/八度——能直接到 ~0.9，但本质是"模板对齐"而非通用转录。
2. 换针对吉他训练的转录模型（如 fretboard-transformer 类）替代 basic-pitch。
3. 接受 octave-insensitive 指标（当前 0.6714–0.7143）作为该素材的合理上限。

## 评测期间新增的回滚保护旋钮（全部默认关闭/向后兼容）

`PolyphonicAudioAnalyzer`: `octave_ratio`, `baseline_percentile`, `novelty_weight`, `freq_weight`, `frontend="stft"`。match_answer.py 有对应 CLI。pytest：153 passed。

## 后续：Basic Pitch 神经后端落地（2026-08-30 第二轮）

- `src/audio/basic_pitch_backend.py`：`BasicPitchAnalyzer`（懒加载模型、兼容垫片 `TF_USE_LEGACY_KERAS=1` + `scipy.signal.gaussian` patch）。
- GUI 新增「神经转录/Basic Pitch（指弹推荐）」模式，默认参数 onset=0.3 / frame=0.15 / min_len=60ms（实测最优）。
- `requirements-neural.txt`：`basic-pitch[onnx]` + `tensorflow` + `tf_keras` + `setuptools<81`（resampy 依赖 pkg_resources）。
- 晴天评测（强制 offset=25.0）：BP(0.3/0.15/60) strict 0.3143 / midi 0.70；BP(0.25/0.12/50) midi 0.7571（最高），strict 0.2714。CQT 仍 0.3857。
- 已知修复：密集神经音符（单组 >6 音）曾使 `FingeringOptimizer._group_assignments` 笛卡尔积爆炸挂死；已改为惰性扫描 + est>3M 时 10 万组合上限。
- 结论维持：strict 差距主要在 TAB 指法/时间映射，而非音高检测；midi_recall 由 CQT 0.6714 → BP 0.7571。

## 第三轮：confidence 过滤（2026-08-30）

`BasicPitchAnalyzer(min_confidence=...)` 在进入指法层前丢弃弱音符：

| 配置 | strict | midi | offset |
|---|---|---|---|
| BP 0.3/0.15/60, conf 0.0 | 0.3143 | 0.7143 | 18.5 |
| **BP 0.3/0.15/60, conf 0.2**（GUI 新默认） | **0.3571** | 0.6571 | 25.0 |
| BP 0.3/0.15/60, conf 0.35 | 0.1857 | 0.5000 | 25.0 |
| BP 0.25/0.12/50, conf 0.15 | 0.3143 | 0.7286 | 18.5 |

过滤显著清理了输出（notes 2481→1761）且把 offset 拉回正确的 25.0，但 strict 未能越过 CQT 基线 0.3857。两轮无进一步提升，按协议停止扫描。
