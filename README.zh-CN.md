# Rin NPU Agent

[English](README.md) | 简体中文

Rin NPU Agent 是一个面向 Snapdragon Android 设备的**端侧 AI Agent / 本地 AI 工作台**。当前项目基于 Qualcomm GenieX 的 Android 本地推理能力继续开发，支持本地 LLM/VLM、项目化多会话、授权工作区文件 Agent，并正在接入 Snapdragon 8 Elite（SM8750）上的 WAI / SDXL NPU 生图路线。

**当前公开 Debug：v1.5 · versionCode 6 · `com.geniex.demo` · 仅 `arm64-v8a`**

## 当前功能

### 对话 / Agent 模式

- 通过 GenieX 运行本地 LLM / VLM。
- 存在兼容 QAIRT 模型时优先 NPU；兼容运行时也可使用 GPU / CPU。
- 持久化 **项目 → 对话 → 消息** 结构。
- 不同项目/会话默认隔离。
- 只有用户明确要求读取其他项目/对话上下文时，才会在该轮临时开放历史读取工具。
- 每个项目可以独立绑定 Android SAF 工作区。
- Agent 支持目录浏览、文件读取/写入、创建目录、本地生成 PPTX 等。
- LLM/VLM 模型权重不内嵌在 APK 中，通过模型库按需下载/选择。

### 生图模式

当前目标路线是 **WAI Illustrious SDXL + SDXL-Lightning 8-step**，面向 Snapdragon 8 Elite / SM8750 的 QNN/HTP NPU 部署。

已经实现：

- 对话模式 / 生图模式切换。
- 正向 Prompt 与 Negative Prompt **完全独立的预设库**。
- 正向预设支持“名称 + 备注 + 完整 Prompt”。
- Negative Prompt 支持独立保存、默认项和锁定。
- 默认 `1024×1024`，并提供多组分辨率预设。
- 稳定 8-step 配置入口。
- 真实解析 `CLIP → UNet 1/8…8/8 → VAE → 保存` 进度。
- runtime 支持时，通过 `preview_current.png` 刷新中间生成预览。
- 生成前可释放聊天模型 native context，为 SDXL 腾出 RAM。
- runtime / context 缺失时明确报错，不使用假进度或假“模型就绪”。

> [!IMPORTANT]
> **SM8750 的 WAI 预编译 QNN context 包目前尚未随仓库/Release 发布。**
>
> 因此 v1.5 APK 虽然已经包含生图 UI、安装器框架和手机端 runtime driver，但实际 NPU 生图仍需要对应的 `sdxl_qnn` context 包。当前点击“安装 NPU 生图模型”时看到“模型包尚未发布”的提示属于预期状态，并不代表 APK 本身安装失败。

## Debug APK

GitHub Release 会附带：

`Rin-NPU-Agent-v1.5-debug-upgrade.apk`

- ABI：`arm64-v8a`
- 包名：`com.geniex.demo`
- 版本：`1.5` / `versionCode 6`
- 内嵌 GGUF/Test Model：0
- SHA-256：`f0439cacc80830baaf0ba7cc7d15c5cc494499d581fc0f779420a302d4e7a764`
- Debug 签名证书 SHA-256：`afeef7cf03c2bc3b411932df89330af1a6416f62e379096d72807305e3d9f801`

这是 Debug 构建，主要用于开发和测试。只有已安装版本使用相同签名证书且 versionCode 更低时，才能直接原位升级。

## Rin Design System：GUI 的 integrity

项目当前的界面不再使用“功能写完后再美化”的方式，而是把 **integrity（整体一致性）**作为开发约束：新增控件如果重新掉回默认 Android 的锋利/割裂样式，就视为功能没有真正完成。

主要设计规则：

- 统一柔和圆角，避免尖锐默认控件。
- 低对比描边和柔和表面层级。
- Chat、Drawer、模型库、生图页共用同一套卡片/输入框/按钮语言。
- Primary / Secondary / Ghost / Destructive 操作层级明确。
- 标题、分区、正文、状态/辅助信息使用克制的文字层级。
- 间距节奏以 `8 / 12 / 16 / 24 dp` 为核心。
- 低层 runtime 数据视觉降级，不让界面呈现“调试工具感”。
- 新控件必须复用集中化的 colors/styles/drawables，避免重新硬编码十六进制颜色或系统 `<Button>`。

整体方向受到 Material 体系启发，但目标不是复制标准 Material 页面，而是更柔和、扁平、连续的 Rin 产品语言。

## 项目结构

```text
Rin NPU Agent
├─ 对话模式
│  ├─ GenieX Model Manager
│  ├─ LLM / VLM Runtime
│  ├─ 项目 → 多对话
│  └─ SAF 项目工作区 Agent
│
└─ 生图模式
   ├─ 正向 Prompt 库
   ├─ Negative Prompt 库
   ├─ 分辨率 / 8-step 配置
   ├─ WAI/SDXL 手机端 runtime driver
   ├─ QNN context/runtime 检查
   └─ 实时进度 + 可选中间预览
```

## 主要目标平台

当前主要开发/验证目标是 **Snapdragon 8 Elite（SM8750）** Android 手机，APK 本身仅包含 ARM64 ABI。

当前构建配置：

- minSdk 31
- targetSdk / compileSdk 34
- Java 17
- AGP 8.13.0
- Kotlin 2.2.0
- NDK 27.3.13750724
- GenieX Android 0.3.5

## 构建

使用 JDK 17、Android SDK 34 和兼容的 Gradle 8.13：

```bash
gradle assembleDebug
```

APK 默认输出：

```text
build/outputs/apk/debug/app-debug.apk
```

仓库不会提交 `local.properties`、SDK/NDK/JDK、本地 Gradle 缓存、模型缓存、build 目录、签名材料、网站部署配置或 SSH 信息。

## 参考/上游工程

### Qualcomm AI Hub Apps / GenieX

项目最初基于 Qualcomm `ai-hub-apps` 中的 `geniex_chat_android`：

- https://github.com/qualcomm/ai-hub-apps
- https://github.com/qualcomm/geniex

这一基线提供了最初的 GenieX Android 接入、模型管理和本地推理结构；当前项目已经在会话体系、Agent、模型管理、生图模式和整体 GUI 上进行了大量扩展/替换。

### Model-To-NPU

手机端 SDXL/QNN 路线参考：

- https://github.com/VitalikDen0/Model-To-NPU

主要参考了 Snapdragon 8 Elite 上的 QNN/HTP SDXL 路线、拆分 UNet context、Lightning 8-step、stdout 真实进度、中间预览和 LoRA context-slot 思路。

本仓库包含的 Model-To-NPU 衍生 runtime driver 仍遵循 **PolyForm Noncommercial License 1.0.0**，并保留其 LICENSE / NOTICE / Required Notice。详见 `THIRD_PARTY.md` 和 `LICENSES/`。

### 模型侧

计划中的生图路线涉及 WAI Illustrious SDXL 与 ByteDance SDXL-Lightning。仓库当前**不再分发这些模型权重或预编译 QNN context**，对应文件仍受各自上游许可证/使用条款约束。

## 许可证说明

这是一个混合上游来源的工程，不能简单认为所有文件都使用同一许可证：

- Qualcomm AI Hub Apps 衍生代码：BSD-3-Clause。
- Model-To-NPU 衍生 runtime：PolyForm Noncommercial 1.0.0。
- 其他依赖和模型遵循各自上游许可。

详见 [`THIRD_PARTY.md`](THIRD_PARTY.md) 和 [`LICENSES/`](LICENSES/)。

## 下一步

- 发布 SM8750 WAI SDXL QNN context 包。
- 完成 context 一键下载 → SHA-256 校验 → 自动安装。
- 接入按需加载的 2B 级 Prompt 扩写模型，并在生图前立即卸载。
- LoRA 目录自动发现、触发词匹配与 QNN context-slot 热切换。
- 后续所有新增页面/控件继续遵守 Rin Design System 的 integrity 约束。

## 声明

这是独立的个人/研究项目，并非 Qualcomm、WAI、ByteDance 或 Model-To-NPU 的官方产品。
