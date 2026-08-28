# GuitarBapu

GuitarBapu 是一款面向吉他手的 AI 吉他扒谱软件，目标是将音频中的吉他演奏转换为可编辑、可导出的音符和六线谱，帮助用户更快学习和整理歌曲。

当前版本已经具备可用的基础单音扒谱工作流：导入音频、检测音高、清理误检、估计节拍、量化时值、选择琴弦/品位并生成文本六线谱。复音歌曲、可视化编辑和文件导出仍在后续阶段。

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

Phase 5（TAB Generator）已经完成。Audio Loader 可读取 MP3、WAV 和 FLAC；分析器会执行基础单音音高检测、短音过滤、同音合并、音高去抖、onset 分割、BPM 估计和 1/16 拍网格量化，随后通过全局指法优化生成可演奏的文本六线谱。

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

### Phase 4 音符清理与节奏

`analysis.raw_notes` 保存清理前事件，`analysis.notes` 是可继续用于指板映射的稳定结果。默认规则：

- 删除短于 0.08 秒的音符；
- 合并间隔不超过 0.08 秒的同音；
- 删除最长 0.06 秒、夹在相同音高之间的抖动；
- onset 分割后的每个片段至少 0.15 秒；
- 休止段最短 0.12 秒；
- 有稳定 BPM 时按每拍四等分（1/16 音符网格）量化。

节奏结果位于 `analysis.rhythm`，其中包含 `timing`、`onset_times`、`quantized_notes` 和 `rests`。无法稳定估计 BPM 时，程序保留原始秒级时间，不会中断音高分析。

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

### TAB 生成示例

```python
from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import load_audio
from src.music.tab_generator import TabGenerator
from src.music.tab_renderer import TextTabRenderer

analysis = AudioAnalyzer().analyze(load_audio("guitar.wav"))
tablature = TabGenerator().generate(analysis)
print(TextTabRenderer().render(tablature))
```

TAB 默认采用 4/4、每拍四等分、每行四小节。指法选择使用动态规划，综合考虑低品位、换把距离、跨弦距离和和弦品位跨度。无法映射的音符会在六条弦上显示 `x`，并在结果末尾给出警告，不会被静默丢弃。

同一拍的多个 `Note` 可以作为结构化和弦分配到不同琴弦，但当前音高分析器仍是单音算法；这项结构能力是为后续复音识别预留的。

## 后续规划

1. Phase 6：增加项目文件、MIDI、MusicXML、文本 TAB 和 PDF 导出。
2. Phase 7：完善波形、播放定位、编辑和导出 GUI。
3. Phase 8/9：音源分离、复音识别、技巧识别和产品化。

## 运行项目

安装依赖后，可以运行 GUI：

```bash
python -m src.gui.app
```

macOS 也可以在 Finder 中双击 `run_app.command`。GUI 会在后台分析，不会阻塞窗口，并显示四小节换行的六线谱、映射统计、原始/清理后音符、BPM、量化时间和置信度。

当前结果适合清晰的吉他单音、音阶和调音录音。完整歌曲、和弦识别、推弦/滑弦等技巧以及图形化编辑不属于当前算法能力。

运行自动化检查：

```bash
python -m pytest
```
