# Rin NPU Agent

[English](README.md)

Rin NPU Agent 是面向 Snapdragon NPU 设备的 Android ARM64 本地 AI 工作空间，将本地 LLM/VLM 对话、项目级 Agent、持久化项目与对话，以及 WAI/SDXL 生图工作台整合在同一个应用中。

**当前稳定版本：** [`1.6.0`](https://github.com/reimurin/Rin-NPU-Agent/releases/tag/v1.6.0) · version code `28` · 包名 `com.geniex.demo` · `arm64-v8a`

[下载 APK](https://github.com/reimurin/Rin-NPU-Agent/releases/download/v1.6.0/Rin-NPU-Agent-v1.6.0-debug-runtime-ready.apk)

## 1.6.0 已实现功能

### 本地对话与 Agent

- 通过 Qualcomm GenieX 运行本地 LLM / VLM。
- 兼容 QAIRT 的模型优先走 NPU；GGUF 模型可使用已支持的 GPU / CPU 路线。
- 持久化 **项目 → 对话 → 消息** 层级。
- 每个项目独立绑定 SAF 工作目录。
- 用户明确要求时，可读取其他项目和其他对话的上下文。
- Agent 文件工具仅在当前项目授权目录内运行。
- Agent 工具链支持本地 PowerPoint 生成。
- 应用内模型库支持模型选择、下载、加载与卸载。
- 模型权重作为本地模型资产管理，不打包进 APK。

### WAI / SDXL 生图

生图工作台使用 **WAI Illustrious SDXL + SDXL-Lightning 8-step**，在 SM8750 / Snapdragon 8 Elite 目标上通过 Qualcomm QNN/HTP 路线运行。

- 当前 SM8750 模型包提供原生 `1024×1024`、`832×1216`、`1216×832` 三种分辨率。
- 正向 Prompt 与负向 Prompt 独立编辑与预设管理。
- 正向预设支持名称、备注与完整 Prompt。
- 负向预设支持默认、复用与锁定。
- 真实生成进度：`CLIP → UNet 1/8 … 8/8 → VAE → 保存`。
- 在选定去噪步骤显示低开销实时预览。
- Application 级生图 Session，在 Activity 重建、前后台切换时保持生成状态。
- 前台生成服务提供持续通知与明确的“停止”入口。
- 回到应用后立即恢复最新生成进度与预览快照。
- 生图前自动完成对话模型与 SDXL runtime 的内存交接。
- 生成结果支持应用内预览与保存到图库。
- 生图 runtime 支持就绪检查与环境自动恢复。
- Qualcomm HTP 性能调度使用 QAIRT PerfInfrastructure/DCVS，根据热状态在 Boost / Balanced 间切换，并在恢复后重新进入 Boost。

### 动态 LoRA

- 本地 `.safetensors` LoRA 目录扫描与管理。
- 使用 `<lora:名称:权重>` 标签，权重可直接编辑。
- 支持在当前 rank 预算内组合多个 LoRA。
- 生成前执行兼容性检查。
- 保存最近一次成功 LoRA 生图记录，包括缩略图、权重、时间与 Seed。
- “插入 / 弹出”操作只修改 Prompt 标签，不删除本地 LoRA 文件。
- 提供动态 LoRA NPU 自测与诊断报告导出。

### 统一模型中心

生图模式中的模型相关功能统一收口到一个 **“模型”** 入口，并分成三层：

- **基础模型**：当前 model-pack 版本、可用原生分辨率、LoRA ABI、远端版本与下载大小。
- **LoRA**：本地 LoRA 文件、Prompt 标签与权重管理。
- **高级**：runtime 完整性检查/修复、NPU 自测与诊断导出。

model-pack 更新链支持：

- 签名 manifest 校验。
- 自动源选择，以及 GitHub / 中国大陆镜像源手动选择。
- 检查更新、安装、忽略版本与下载进度显示。
- 每个版本记录分辨率与 LoRA 兼容性元数据。
- 应用升级后继续复用已安装且兼容的 model-pack。

## 模型包系统

固定模型目录为 [`models/index.json`](models/index.json)，Android 端使用固定索引地址：

```text
https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/index.json
```

每个模型包记录目标芯片、ABI、runtime、原生分辨率、压缩包校验值与 GitHub Release 资产。APP 根据当前设备匹配对应的已发布模型包。

大型 QNN context 通过 GitHub Release 资产分发，并按顺序拆分为多个分卷。安装器逐卷下载、执行 SHA-256 校验、校验连续分卷组成的完整压缩流并直接解压。CLIP-G 等组件级修复可以独立安装。

当前 SM8750 model-pack 提供 1.6.0 使用的 `1024×1024`、`832×1216`、`1216×832` 三条原生生图路线。

## 应用结构

```text
Rin NPU Agent
├─ 对话模式
│  ├─ GenieX 模型管理
│  ├─ 本地 LLM / VLM runtime
│  ├─ 项目 → 对话 → 消息
│  ├─ 项目级 SAF 工作区
│  └─ 本地 Agent 工具
│
└─ 生图模式
   ├─ 正向 / 负向 Prompt 预设
   ├─ 原生分辨率选择
   ├─ WAI / SDXL-Lightning 8-step
   ├─ Application 级生成 Session
   ├─ 前台生成服务
   ├─ 实时预览 + 进度快照
   ├─ HTP 性能调度
   ├─ 动态 LoRA 管理
   └─ 模型中心
      ├─ 基础模型
      ├─ LoRA
      └─ 高级诊断
```

## Rin Design System

应用在对话、Drawer、生图、模型管理、预设和弹窗中统一使用 **Rin Design System**。

- 统一柔和圆角。
- 低对比描边与表面层级。
- 卡片、输入框、按钮、列表和弹窗共享视觉体系。
- 主操作 / 次级 / Ghost / 危险操作明确分层。
- 标题、分区标题、正文、辅助信息统一层级。
- `8 / 12 / 16 / 24 dp` 间距节奏。
- 颜色、样式和 drawable 集中管理。
- 新增控件与页面保持 UI integrity。

## 目标平台

当前主要验收目标：**Snapdragon 8 Elite / SM8750** Android 设备。

- `minSdk 31`
- `targetSdk 34`
- `compileSdk 34`
- Java 17
- Android Gradle Plugin 8.13.0
- Kotlin 2.2.0
- NDK 27.3.13750724
- GenieX Android 0.3.5

1.6.0 的生图与 LoRA 主链已在 Snapdragon 8 Elite 的荣耀 Magic 7 Pro 上完成真机验收。

## 构建

先按 [`native/README.md`](native/README.md) 准备 native 依赖，再使用 JDK 17、Android SDK 34 与 Gradle 8.13 构建：

```bash
gradle testDebugUnitTest assembleDebug
```

APK 输出：

```text
build/outputs/apk/debug/app-debug.apk
```

本地 SDK / NDK / JDK、缓存、模型文件、签名材料与部署配置独立存放，不进入仓库。

## 上游工程与参考

### Qualcomm AI Hub Apps / GenieX

Rin NPU Agent 最初基于 Qualcomm `geniex_chat_android`，并继续使用 GenieX 作为本地模型运行与模型管理层：

- https://github.com/qualcomm/ai-hub-apps
- https://github.com/qualcomm/geniex

### Model-To-NPU

SDXL/QNN 手机端 runtime 延续并扩展了 Model-To-NPU 的相关实现：

- https://github.com/VitalikDen0/Model-To-NPU

包括 Snapdragon 8 Elite QNN/HTP SDXL 路线、拆分 UNet context、Lightning 8-step、runtime 进度协议、实时预览与 QNN LoRA 路线。

项目中来自 Model-To-NPU 的 runtime 文件继续保留 PolyForm Noncommercial 1.0.0 与 Required Notice，详见 [`THIRD_PARTY.md`](THIRD_PARTY.md) 和 [`LICENSES/`](LICENSES/)。

### 模型侧

生图路线使用 WAI Illustrious SDXL 与 ByteDance SDXL-Lightning。预编译模型包通过模型目录与 GitHub Releases 分发。

## 许可

- Qualcomm AI Hub Apps 衍生代码：BSD-3-Clause。
- Model-To-NPU 衍生 runtime 组件：PolyForm Noncommercial 1.0.0。
- 其他依赖与模型包遵循各自上游许可。

详见 [`THIRD_PARTY.md`](THIRD_PARTY.md) 和 [`LICENSES/`](LICENSES/)。
