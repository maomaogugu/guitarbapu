# GuitarBapu

GuitarBapu 是一款面向吉他手的 AI 吉他扒谱软件，目标是将音频中的吉他演奏转换为可编辑、可导出的音符和六线谱，帮助用户更快学习和整理歌曲。

当前版本已经具备可用的基础单音扒谱工作流：导入和播放音频、查看波形、检测音高、清理误检、估计节拍、量化时值、选择琴弦/品位、生成并编辑六线谱，以及保存项目或导出 TXT、MIDI 和 MusicXML。复音歌曲和高级演奏技巧识别仍在后续阶段。

## 项目结构

```text
.
├── src/
│   ├── audio/          # 音频加载、录音和分析边界
│   ├── exporters/      # 文本 TAB、MIDI 和 MusicXML 导出
│   ├── music/          # 音符、吉他和六线谱模型
│   ├── project/        # 版本化项目文件与 JSON 持久化
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

Phase 7（GUI 工作流）已经完成。Audio Loader 可读取 MP3、WAV 和 FLAC；分析器会执行基础单音音高检测、短音过滤、同音合并、音高去抖、onset 分割、BPM 估计和 1/16 拍网格量化，随后通过全局指法优化生成可演奏的文本六线谱。桌面界面支持音频播放、波形定位、循环选区、TAB 事件编辑、撤销/重做、项目持久化以及 TXT、MIDI 和 MusicXML 导出。

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

### Phase 6 项目保存与导出

GUI 在分析成功后启用以下操作：

- 保存项目：写入版本化的 `.guitarbapu.json` 文件；
- 打开项目：不重新分析音频，直接恢复音符、节奏、调弦和 TAB；
- 导出 TAB：生成与界面显示一致的 UTF-8 文本六线谱；
- 导出 MIDI：保留 MIDI 音高、力度、拍位置、时值、BPM 和拍号；
- 导出 MusicXML：额外写入琴弦和品位技术标记，可由 MuseScore 等软件读取。

项目文件只保存原音频的相对路径引用，不嵌入波形数据，也不保存体积较大的逐帧 `pitch_hz` 数组。因此项目文件较小；即使原音频移动或丢失，仍然可以打开并查看、导出已有 TAB，但不能直接重新播放或重新分析该音频。

也可以直接调用导出 API：

```python
from src.exporters import export_midi, export_musicxml, export_text_tab
from src.project import TranscriptionProject, save_project

project = TranscriptionProject(
    audio_path="guitar.wav",
    analysis=analysis,
    tablature=tablature,
)
save_project(project, "guitar.guitarbapu.json")
export_text_tab(tablature, "guitar.txt")
export_midi(tablature, "guitar.mid")
export_musicxml(tablature, "guitar.musicxml")
```

当前未实现 PDF 和 Guitar Pro 原生导出。MusicXML 是现阶段与外部制谱软件交换数据的推荐格式；MIDI 不保存吉他专属排版，文本 TAB 则保留当前选择的琴弦和品位。

### Phase 7 播放、波形和 TAB 编辑

导入音频或打开带有有效音频引用的项目后，可以：

- 播放、暂停、停止和拖动时间滑块；
- 点击波形跳转，或在波形上拖动选择循环区间；
- 点击 TAB 事件定位到对应音频，并将该事件作为循环选区；
- 查看音符、MIDI、琴弦、品位、开始拍、时值、小节、技巧和可信度；
- 插入、修改和删除 TAB 事件；
- 修改 MIDI 音高时自动选择接近原位置的可演奏指法；
- 从该音高的合法指板位置中切换琴弦和品位；
- 插入和弦或调整时间时避免同一琴弦上的音符重叠，并尝试其他合法位置；
- 修改开始拍、时值以及 slide、hammer-on、pull-off、bend、vibrato 等技巧标签；
- 撤销、重做，并在离开存在未保存修改的项目时选择保存、放弃或取消。

编辑逻辑位于独立的 `TabEditController`，GUI 不直接计算指板音高。导出和项目保存使用编辑后的 `Tablature`；`AudioAnalysis.raw_notes` 和清理后的分析音符继续保留为算法来源记录。

分析仍在后台线程中执行，窗口不会被冻结。“取消分析”是安全取消：程序会丢弃结果并在当前 librosa 调用结束后释放任务，无法在 YIN 内部计算进行到一半时强制终止。

## 后续规划

1. Phase 8：增加可选的吉他音源分离和多轨分析。
2. Phase 9：复音识别、技巧识别、安装包和产品化。

## 运行项目

安装依赖后，可以运行 GUI：

```bash
python -m src.gui.app
```

macOS 也可以在 Finder 中双击 `run_app.command`。GUI 会在后台分析，不会阻塞窗口，并显示波形、播放游标、四小节换行的六线谱以及可编辑事件表。分析完成后可选择事件定位音频、循环练习、修改音高/指法/时值/技巧、撤销重做、保存项目，或导出 TAB、MIDI 和 MusicXML。

当前结果适合清晰的吉他单音、音阶和调音录音。完整歌曲、和弦识别、推弦/滑弦等技巧以及图形化编辑不属于当前算法能力。

运行自动化检查：

```bash
python -m pytest
```
