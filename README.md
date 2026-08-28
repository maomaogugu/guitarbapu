# GuitarBapu

GuitarBapu 是一款面向吉他手的 AI 吉他扒谱软件，目标是将音频中的吉他演奏转换为可编辑、可导出的音符和六线谱，帮助用户更快学习和整理歌曲。

当前版本已经具备可用的基础单音扒谱工作流，并增加了实验性和弦/复音、逻辑多轨和演奏技巧候选：导入和播放音频、可选分离吉他 stem、检测音高与和弦、将事件分入主音/节奏候选轨、识别滑弦/击勾弦/推弦/颤音候选、生成并编辑六线谱，以及保存项目或导出 TXT、MIDI 和 MusicXML。真正的吉他内部音频分轨和高精度技巧模型仍属于后续增强。

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

如果需要 Phase 8 的吉他音源分离，建议使用 Python 3.12 的虚拟环境安装可选依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-separation.txt
```

基础功能不依赖 PyTorch/Demucs；不安装时 GUI 会禁用“先分离吉他”，仍可直接分析原音频。

运行当前骨架的冒烟测试：

```bash
python -m pytest
```

## 当前开发阶段

Phase 9D（产品化基础）已完成基础版。除 Phase 9A–9C 的分析和编辑能力外，软件现在提供跨平台启动脚本、系统诊断、滚动日志、崩溃报告、可选 Demucs 模型状态检查/显式下载入口，以及 PyInstaller 打包配置。

### 音高检测示例

```python
from src.audio.analyzer import AudioAnalyzer
from src.audio.loader import load_audio

audio = load_audio("guitar.wav")
analysis = AudioAnalyzer().analyze(audio)
for note in analysis.notes:
    print(note.name, note.start, note.duration)
```

`analysis.features["pitch_hz"]` 保存逐帧的频率数组；静音或低能量帧以 `NaN` 表示。检测器默认保留标准吉他可演奏范围 E2-E6（`min_note_midi=40`、`max_note_midi=88`），避免把低于六弦空弦的假低音写进 TAB；其他调弦可通过 `min_note_midi`、`max_note_midi`、`fmin_hz`、`fmax_hz`、`frame_length` 和 `hop_length` 调整。

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

同一拍的多个 `Note` 可以作为结构化和弦分配到不同琴弦。默认 YIN 模式仍为稳定的单音算法；Phase 9A 实验模式可以向该结构提供同时音符。

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

### Phase 8 可选吉他音源分离

导入完整歌曲后，勾选“先分离吉他（Demucs）”再开始分析。程序会：

1. 使用 `htdemucs_6s` 模型分离它明确提供的 `guitar` stem；
2. 将吉他 stem 交给现有 Audio Loader、Pitch Detection 和 TAB Generator；
3. 在 GUI 中提供“原音频 / 吉他分离轨”播放切换；
4. 按原始音频内容和模型配置缓存结果，同一文件再次分析可直接复用。

macOS 缓存默认位于 `~/Library/Caches/GuitarBapu/separation`，不在 Git 仓库内。项目文件仍保存原音频路径，只额外记录模型、设备和缓存键；重新打开项目时，若缓存尚在，会恢复吉他分离轨播放。

为什么需要下载模型：Demucs 不是一组固定的音频规则，而是经过大量混音/分轨数据训练后得到的神经网络权重。Python 包只包含运行代码，模型权重需要首次单独下载，之后会从用户缓存复用。

已知限制：

- 当前 Mac 上 PyTorch MPS 不可用时会自动使用 CPU，完整歌曲可能较慢；
- 模型首次下载和初次加载不一定能立即取消；
- `guitar` stem 可能仍有人声、鼓或其他乐器泄漏，不保证完全干净；
- 当前只保存请求的吉他 stem，不会把 `other` 假定为吉他；
- 数据对象已支持多 stem 扩展，但当前不会自动拆分主音吉他和节奏吉他；
- 分离只负责提取吉他 stem；选择单音或实验性复音分析后，结果仍受后续音高算法准确率限制。

### Phase 9A 实验性和弦/复音识别

GUI 的“分析模式”提供两个选项：

- **单音（推荐）**：使用现有 YIN 流程，适合 Solo、音阶、单音 riff 和调音录音；
- **和弦/复音（实验）**：使用 CQT 频谱、onset 分段、相对能量阈值和泛音抑制，每个时段最多保留 6 个吉他音高。

实验模式目前识别 major、minor、power chord、sus2、sus4、diminished 和 augmented 的基础标签。标签无法确定时仍保留检测到的具体 MIDI 音高，不会丢弃数据。

```python
from src.audio.loader import load_audio
from src.audio.polyphonic_analyzer import PolyphonicAudioAnalyzer
from src.music.tab_generator import TabGenerator

analysis = PolyphonicAudioAnalyzer().analyze(load_audio("clean-chords.wav"))
for chord in analysis.chords:
    print(chord.name, chord.midis, chord.start, chord.confidence)

tablature = TabGenerator().generate(analysis)
```

复音结果会保存到 `.guitarbapu.json`，旧项目没有 `chords` 字段时仍可正常打开。指法优化器会限制每个和弦的候选指法数，防止连续和弦产生组合爆炸。

已知限制：

- 这是无新增大型模型的确定性基线，不是经训练的高精度复音 AI；
- 干净、延音较稳定的和弦效果最好，快速闷音、强失真和密集混音可能误检；
- 吉他泛音和真实高八度音在频谱上可能难以区分；
- 和弦转位和根音存在听觉歧义，当前优先尝试以最低检测音为根音；
- 还不支持七和弦、九和弦、分数和弦或乐队级和声分析；
- 实验结果应在 GUI 中试听并人工修正。

### Phase 9B 逻辑主音/节奏多轨

分析完成后，GUI 会根据事件结构生成一条或多条逻辑轨道：

- **主音候选**：同一时段只有一个检测音高；
- **节奏候选**：同一时段有两个以上音高或存在 `Chord`；
- **未分类**：事件可信度低于分类阈值；
- **Solo**：暂不自动判断，可在 GUI 中将主音候选轨手动改为 Solo。

同一音符事件只会分配到一条逻辑轨，不会在主音和节奏轨中重复。轨道切换时，各自的 TAB、编辑历史、撤销/重做状态和人工角色会独立保留。导出 TAB、MIDI 或 MusicXML 时只导出 GUI 中当前选中的轨道，文件名默认加入 `lead` / `rhythm` / `solo` 后缀。

`.guitarbapu.json` 现在可选保存 `tracks` 和 `active_track_id`；没有这些字段的旧单轨项目仍使用原有 `analysis + tablature` 打开。为了兼容旧版，项目根节点仍保留完整分析/TAB 快照。

重要限制：这些是共享同一原音频或 `guitar` stem 的“音乐事件轨”，不是独立的 `lead-guitar.wav` 和 `rhythm-guitar.wav`。`htdemucs_6s` 只提供一条综合吉他 stem，因此当主音和节奏同时演奏时，本阶段不能将它们的音频完全分开。

### Phase 9C 演奏技巧候选

Phase 9C 使用独立的 `TechniqueAnalyzer` 和 `TechniqueDetection` 数据对象，不把识别规则写入 GUI，也不增加新的大型模型。当前基础规则为：

- **滑弦 `slide`**：相邻音符之间存在持续、方向一致的中间音高轨迹；
- **击弦 `hammer-on`**：音高向上跳转，但目标音缺少明显的第二次振幅起音；
- **勾弦 `pull-off`**：音高向下跳转，但目标音缺少明显的第二次振幅起音；
- **推弦 `bend`**：单个延音内出现持续向上的半音变化；
- **颤音 `vibrato`**：延音内出现约 3.5–9 Hz 的稳定周期性音高调制。

主流程通过 `TranscriptionService` 自动执行技巧识别：

```python
from src.audio.transcription_service import TranscriptionService

result = TranscriptionService().transcribe("guitar.wav")
for detection in result.analysis.techniques:
    print(
        detection.technique.value,
        detection.note.name,
        detection.confidence,
    )
```

单独调用 `AudioAnalyzer().analyze(audio)` 仍只负责基础音高和节奏；这种分层设计使技巧算法可以独立测试或替换，而不改变 Audio Loader 和基础分析器的公开职责。

转换技巧会标记在目标音符上；每个音符当前只保留一个主要技巧候选。`AudioAnalysis.techniques` 保存算法证据，`TabEvent.technique` 和 `technique_confidence` 用于 GUI、项目文件和导出。文本 TAB 会追加技巧候选清单，MusicXML 会保留可读的技巧文字标记；MIDI 格式本身不保存这些吉他专属标签。旧项目没有 `techniques` 或 `technique_confidence` 字段时仍可正常打开。

自动技巧标签必须试听确认。人工在事件编辑器中修改技巧后，自动技巧可信度会被清除，以区分“算法候选”和“用户确认”。

当前限制：

- 只在单音区段自动判断，和弦与重叠音符不会强行分类；
- 强失真、混响、背景乐器泄漏、快速连奏和不稳定音高可能造成漏检或误检；
- 当前不判断滑弦具体方向符号、推弦精确目标品位、预推弦、释放推弦或组合技巧；
- 同一个音符暂时不能同时保存例如“推弦 + 颤音”两个自动标签；
- 这是可解释的确定性基线，不是训练完成的高精度吉他技巧 AI 模型。

### Phase 9D 产品化基础

帮助菜单现在提供：

- **系统诊断**：检查 Python、核心依赖、可选 Demucs 依赖、模型缓存和用户目录；报告可复制或保存，不包含环境变量、音频或项目内容；
- **准备 Demucs 模型**：只有用户明确确认后才下载 `htdemucs_6s` 所需权重，并在后台校验缓存；基础版不安装 Demucs 时仍可正常使用；
- **打开日志目录**：日志默认按平台写入用户日志目录，单个文件滚动保留，便于定位导入、分析、导出和模型错误。

未捕获的主线程或后台线程异常会生成带时间戳的文本崩溃报告。日志、模型和分离缓存都位于用户目录，不会写入仓库。

源码运行方式：

```bash
./run_app.sh                 # macOS / Linux
./run_app.command             # macOS Finder 双击
run_app.bat                  # Windows
```

本地打包需要先安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
python scripts/build_app.py                 # 基础包，不含 PyTorch/Demucs
python scripts/build_app.py --with-separation  # 含可选分离，体积明显更大
```

产物位于 `dist/GuitarBapu/`。打包必须在目标操作系统上执行；当前配置是可复现的起点，不等同于已经签名、公证或发布到应用商店的安装包。Full 版会把 PyTorch 和 Demucs 一并打包，但模型权重仍建议由用户在应用内显式下载，以避免把大文件提交到 Git。

## 后续规划

1. 用有技巧标注的自有或授权 DI 吉他素材评估 Phase 9C 的精确率、召回率和时间边界。
2. 在 macOS、Windows、Linux 各生成并实测基础包和 Full 包，补充签名、公证、安装器和自动更新。
3. 根据评测结果调整技巧阈值，并决定是否引入专用轻量技巧模型。

## 运行项目

安装依赖后，可以运行 GUI：

```bash
python -m src.gui.app
```

macOS 也可以在 Finder 中双击 `run_app.command`。GUI 会在后台分析，不会阻塞窗口，并显示波形、播放游标、四小节换行的六线谱以及可编辑事件表。分析完成后可选择事件定位音频、循环练习、修改音高/指法/时值/技巧、撤销重做、保存项目，或导出 TAB、MIDI 和 MusicXML。

`run_app.command` 和 `run_app.sh` 会优先使用项目中的 `.venv/bin/python`，`run_app.bat` 会优先使用 `.venv\\Scripts\\python.exe`。因此可选 Demucs 依赖安装到对应虚拟环境后，启动脚本能正确识别。

当前单音模式适合清晰的吉他 Solo、音阶和技巧测试录音；实验性复音模式适合干净、稳定的和弦。技巧识别属于需要试听确认的实验功能，真实混音的高精度复音/技巧识别和图形化编辑仍属于后续能力。

运行自动化检查：

```bash
python -m pytest
```

如果手边有已获授权或自制的参考 TAB，可以用本地评测脚本比较程序输出；参考谱默认不提交进仓库：

```bash
python scripts/compare_ascii_tab.py song.mp3 ref.txt --mode polyphonic
GB_EVAL_CASES=eval_cases.local.json GB_EVAL_MAX_RUNS=1 ./scripts/run_eval_loop.sh
```
