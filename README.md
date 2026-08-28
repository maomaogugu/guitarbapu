# GuitarBapu

GuitarBapu 是一款面向吉他手的 AI 吉他扒谱软件。当前仓库处于项目骨架阶段，先完成模块边界和开发基础设施，暂未实现具体扒谱功能。

## 项目结构

```text
.
├── src/
│   ├── audio/          # 音频文件加载、麦克风录音与预处理
│   ├── music/          # 音符、和弦、节奏与六线谱模型
│   ├── gui/            # 桌面应用界面与交互入口
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

运行当前骨架的冒烟测试：

```bash
python -m pytest
```

## 当前状态

- 已建立音频、音乐模型、GUI、工具和测试目录。
- 已提供各模块的 Python 包入口与后续实现占位。
- 已加入音频处理、音乐表示、桌面 GUI 和测试所需的基础依赖清单。

## 后续开发建议

1. 先确定音频输入格式、采样率和录音生命周期，完成 `src/audio` 的统一数据接口。
2. 建立音符、节拍、和弦及六线谱的数据模型，再接入音高与节奏识别算法。
3. 以可编辑谱面为核心设计 GUI，最后补充导出、项目保存和端到端测试。
