# GuitarBapu

GuitarBapu 是一款面向吉他手的 AI 吉他扒谱软件，目标是将音频中的吉他演奏转换为可编辑、可导出的音符和六线谱，帮助用户更快学习和整理歌曲。

当前版本只完成可扩展的基础架构，不包含 AI 扒谱算法或可用的桌面界面。

## 项目结构

```text
.
├── src/
│   ├── audio/          # 音频加载、录音和分析边界
│   ├── music/          # 音符、吉他和六线谱模型
│   ├── gui/            # 桌面应用入口
│   └── utils/          # 配置、日志及跨模块工具
├── tests/              # 自动化测试
├── requirements.txt    # 运行与开发依赖
└── README.md
```

## 开发环境

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

桌面界面基于 PyQt6 构建。

运行当前骨架的冒烟测试：

```bash
python -m pytest
```

## 当前开发阶段

当前处于 Phase 3（Guitar Fretboard Mapping）：Phase 2 已完成基础单音音高检测，Audio Loader 可读取 MP3、WAV 和 FLAC。`Fretboard` 会将 `Note.midi` 映射到标准吉他的所有可用琴弦/品位，并提供优先低品位、减少换把的基础选择策略。完整的复音扒谱、节奏识别和 GUI 工作流仍是后续实现点。

### 音高检测示例

```python
from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import load_audio

audio = load_audio("guitar.wav")
analysis = AudioAnalyzer().analyze(audio)
for note in analysis.notes:
    print(note.name, note.start, note.duration)
```

`analysis.features["pitch_hz"]` 保存逐帧的频率数组；静音或低能量帧以 `NaN` 表示。检测器默认覆盖约 C2-C6，可通过 `fmin_hz`、`fmax_hz`、`frame_length` 和 `hop_length` 调整。

### 指板映射示例

```python
from src.music.fretboard import Fretboard
from src.music.note import Note

fretboard = Fretboard()
note = Note(midi=64)                     # E4
positions = fretboard.find_positions(note)  # 所有可用位置
chosen = fretboard.choose_position(note)    # 基础低品位策略
```

标准吉他默认是 E-A-D-G-B-E、24 品；`Guitar.standard(capo=2)` 可表示使用变调夹的情况。不可演奏的 MIDI 音符会返回空位置列表，连续音符可通过 `map_notes()` 选择较平滑的指法。当前策略是启发式规则，尚未处理手指编号、和弦按法或复杂技巧。

## 后续规划

1. 固定音频输入格式、采样率和录音生命周期，接入 `src/audio` 的加载、录音和预处理实现。
2. 完善节拍、和弦、音符到琴弦/品位的音乐模型，再接入复音音高与节奏识别算法。
3. 构建可编辑的 Qt 六线谱界面，支持播放定位、修改和常见格式导出。
4. 增加端到端样例、持久化项目文件和跨平台打包。

## 运行项目

安装依赖后，可以运行当前 GUI 入口（它会显示架构阶段提示）：

```bash
python -m src.gui.app
```

运行自动化检查：

```bash
python -m pytest
```
