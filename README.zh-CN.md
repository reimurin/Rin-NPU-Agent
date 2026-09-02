# Rin NPU Agent

面向 Snapdragon NPU 的 Android ARM64 本地 AI 工作空间，将本地 LLM/VLM 对话、项目级 Agent、多项目多会话，以及 WAI/SDXL 生图工作台整合在同一个应用中。

**当前开发版本线：** `1.5.1` · 包名 `com.geniex.demo` · `arm64-v8a`

## 主要功能

### 对话与 Agent

- 通过 Qualcomm GenieX 运行本地 LLM / VLM。
- 有可用 QAIRT 模型时优先 NPU，同时保留兼容的 GPU / CPU 路线。
- 持久化 **项目 → 对话 → 消息** 层级。
- 每个项目独立绑定 SAF 工作目录。
- 用户明确要求时可读取其他项目/对话上下文。
- 本地文件工具与本地 PowerPoint 生成。
- 模型库负责下载与选择，LLM/VLM 权重不塞进 APK。

### 生图模式

当前生图路线是 **WAI Illustrious SDXL + SDXL-Lightning 8-step + Qualcomm HTP/QNN**。

- 对话 / 生图模式切换。
- 正向 Prompt 与负向 Prompt 独立预设库。
- 正向预设支持名称、备注和完整 Prompt。
- 负向预设支持默认、复用与锁定。
- 分辨率预设，默认 `1024×1024`。
- 真实生成进度：`CLIP → UNet 1/8 … 8/8 → VAE → 保存`。
- runtime 提供预览解码时显示 `preview_current.png` 中间过程。
- 对话模型与 SDXL 之间进行内存交接。
- 根据手机硬件自动选择预编译 NPU 生图包。

首个 **SM8750** WAI 预编译包正在制作。发布后会进入仓库模型目录，手机端自动识别并安装。

## 模型包目录

固定索引文件：[`models/index.json`](models/index.json)。安卓端固定读取：

```text
https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/index.json
```

每个包记录兼容芯片、ABI、runtime、分辨率、压缩包 SHA-256 与 GitHub Release 资产。APP 根据当前手机 SoC/ABI 自动选择最匹配的已发布包。

QNN context 体积较大，模型文件放在 **GitHub Release**，不进入 Git 历史。GitHub Release 单个资产需小于 2 GiB，因此大包按顺序分卷。手机端逐卷下载、逐卷 SHA-256 校验，再重组 ZIP、校验完整包并安装。

以后增加新的 Snapdragon 芯片、分辨率或 QNN profile，只需要发布新模型包并更新 `models/index.json`，APK 的目录地址保持不变。

## 版本命名

之后统一使用语义化版本：

- `1.5.1 / 1.5.2 ...`：修复和小型功能迭代。
- `1.6.0 / 1.7.0 ...`：较大的功能阶段。
- `2.0.0`：未来存在明显兼容性变化的大版本。

Android `versionCode` 继续单调递增，保证原位升级。

## Rin Design System

GUI 使用 **Rin Design System**，其中 **integrity（整体一致性）** 是长期设计约束：新增功能从一开始就使用现有设计语言。

- 统一柔和圆角。
- 低对比描边与表面层级。
- 卡片、输入框、按钮、列表和弹窗共享视觉体系。
- 主操作 / 次级 / Ghost / 危险操作明确分层。
- 标题、分区标题、正文、辅助信息统一层级。
- `8 / 12 / 16 / 24 dp` 间距节奏。
- 状态与 runtime 信息保持克制。
- 颜色、样式和 drawable 集中管理。

整体方向以扁平、柔和、连续为主，在 Material 基础上保持统一的产品感。

## 参考与上游工程

### Qualcomm AI Hub Apps / GenieX

项目最初以 Qualcomm `geniex_chat_android` 为基础，并继续使用 GenieX 作为本地模型运行和模型管理层：

- https://github.com/qualcomm/ai-hub-apps
- https://github.com/qualcomm/geniex

### Model-To-NPU

SDXL/QNN 手机端路线参考并整合了 Model-To-NPU：

- https://github.com/VitalikDen0/Model-To-NPU

包括 Snapdragon 8 Elite QNN/HTP 的 SDXL 路线、拆分 UNet context、Lightning 8-step、runtime 进度协议、中间预览以及 QNN LoRA context slot 等方向。

项目中来自 Model-To-NPU 的 runtime 文件继续保留 PolyForm Noncommercial 1.0.0 与 Required Notice，详见 [`THIRD_PARTY.md`](THIRD_PARTY.md) 和 [`LICENSES/`](LICENSES/)。

### 模型侧

生图路线使用 WAI Illustrious SDXL 与 ByteDance SDXL-Lightning。预编译模型包通过模型目录与 GitHub Release 独立发布。

## 下一步

- 发布首个 SM8750 WAI SDXL QNN 模型包。
- 增加更多分辨率 bucket 和 Snapdragon 芯片目标。
- 加入按需加载的小型 Prompt 扩写模型，生成后立即卸载再进入 SDXL。
- 根据触发词自动发现 LoRA，并切换 QNN context slot。
- 后续页面和控件继续遵守 Rin Design System 的 integrity 规则。
