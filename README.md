# Rin NPU Agent

[简体中文](README.zh-CN.md)

Rin NPU Agent is an Android ARM64 on-device AI workspace for Snapdragon NPU devices. It combines local LLM/VLM chat, project-scoped Agent tools, persistent projects and conversations, and a WAI/SDXL image-generation workspace in one application.

**Stable release:** [`1.6.0`](https://github.com/reimurin/Rin-NPU-Agent/releases/tag/v1.6.0) · version code `28` · package `com.geniex.demo` · `arm64-v8a`

[Download APK](https://github.com/reimurin/Rin-NPU-Agent/releases/download/v1.6.0/Rin-NPU-Agent-v1.6.0-debug-runtime-ready.apk)

## What ships in 1.6.0

### Local Chat & Agent

- Local LLM and VLM inference through Qualcomm GenieX.
- NPU-first execution for compatible QAIRT models, with supported GPU/CPU runtimes available for GGUF models.
- Persistent **Project → Conversation → Message** hierarchy.
- Independent SAF workspace binding for each project.
- Explicit cross-project and cross-conversation context retrieval when requested by the user.
- Local Agent file tools inside the selected project workspace.
- Local PowerPoint generation through the Agent tool chain.
- In-app model library, model selection, downloading, loading and unloading.
- Model weights stay outside the APK and are managed as local model assets.

### WAI / SDXL image generation

The image workspace runs **WAI Illustrious SDXL + SDXL-Lightning 8-step** through the Qualcomm QNN/HTP path on the SM8750 / Snapdragon 8 Elite target.

- Native `1024×1024`, `832×1216`, and `1216×832` generation paths with the current SM8750 model pack.
- Positive and negative prompt editors with reusable preset libraries.
- Named positive presets with notes.
- Default, reusable and lockable negative presets.
- Real pipeline progress: `CLIP → UNet 1/8 … 8/8 → VAE → save`.
- Low-overhead live preview at selected denoising steps.
- Application-level generation session that survives Activity recreation and foreground/background transitions.
- Foreground generation service with persistent notification and explicit Stop action.
- Immediate progress/preview snapshot restoration when returning to the app.
- Memory handoff between the chat runtime and the SDXL runtime before image generation.
- Generated-image preview and gallery saving.
- Runtime readiness checks and automatic recovery for the packaged image-generation environment.
- Qualcomm HTP performance governor using QAIRT PerfInfrastructure/DCVS, with Boost/Balanced thermal switching and recovery back to Boost.

### Dynamic LoRA

- Local `.safetensors` LoRA catalog for the WAI/Illustrious image path.
- Prompt tags in the form `<lora:name:weight>` with directly editable weights.
- Multiple LoRA selections within the supported rank budget.
- Compatibility checks before generation.
- Recent successful LoRA result history with thumbnail, weight, time and seed.
- Insert/eject controls that modify prompt tags without deleting local LoRA files.
- Dynamic LoRA NPU self-test and diagnostic report export.

### Unified Model Center

The image-mode model workflow is consolidated into one **Model** entry with three sections:

- **Basic Model** — installed model-pack version, available native resolutions, LoRA ABI, remote update information and package size.
- **LoRA** — local LoRA catalog and prompt-weight management.
- **Advanced** — runtime integrity check/repair, NPU self-test and diagnostic export.

Model-pack updates include:

- Signed manifest verification.
- Automatic source selection plus explicit GitHub / China-mirror selection.
- Update check, install, ignore-version and progress reporting.
- Per-release resolution and LoRA compatibility metadata.
- Reuse of an already installed compatible model pack across app updates.

## Model package system

The canonical catalog is [`models/index.json`](models/index.json). Android clients use the fixed catalog endpoint:

```text
https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/index.json
```

Each package entry records the target chipset, ABI, runtime, native resolutions, archive checksums and GitHub Release assets. The app matches the current device against the catalog and selects the corresponding published package.

Large QNN context bundles are distributed as GitHub Release assets and split into ordered parts. The installer downloads the parts, verifies SHA-256, validates the combined archive stream, and extracts the package directly. Component-level repairs such as CLIP-G can be installed independently.

The current SM8750 model-pack line provides the native `1024×1024`, `832×1216`, and `1216×832` image-generation paths used by 1.6.0.

## App architecture

```text
Rin NPU Agent
├─ Chat mode
│  ├─ GenieX model manager
│  ├─ Local LLM / VLM runtime
│  ├─ Projects → Conversations → Messages
│  ├─ Project-scoped SAF workspace
│  └─ Local Agent tools
│
└─ Image mode
   ├─ Positive / negative prompt presets
   ├─ Native resolution selector
   ├─ WAI / SDXL-Lightning 8-step pipeline
   ├─ Application-level generation session
   ├─ Foreground generation service
   ├─ Live preview + progress snapshots
   ├─ HTP performance governor
   ├─ Dynamic LoRA manager
   └─ Model Center
      ├─ Basic Model
      ├─ LoRA
      └─ Advanced diagnostics
```

## Rin Design System

The application follows **Rin Design System** across Chat, Drawer, Image mode, model management, presets and dialogs.

- Unified rounded geometry.
- Soft, low-contrast strokes and layered surfaces.
- Shared card, input, button, list and dialog hierarchy.
- Primary / secondary / ghost / destructive action levels.
- Consistent title / section / body / caption typography.
- `8 / 12 / 16 / 24 dp` spacing rhythm.
- Centralized colors, styles and drawables.
- UI integrity across newly added controls and screens.

## Target platform

Primary validated target: **Snapdragon 8 Elite / SM8750** Android devices.

- `minSdk 31`
- `targetSdk 34`
- `compileSdk 34`
- Java 17
- Android Gradle Plugin 8.13.0
- Kotlin 2.2.0
- NDK 27.3.13750724
- GenieX Android 0.3.5

The 1.6.0 image-generation and LoRA chain has been validated on an Honor Magic 7 Pro with Snapdragon 8 Elite.

## Build

Prepare the native prerequisites described in [`native/README.md`](native/README.md), then build with JDK 17, Android SDK 34 and Gradle 8.13:

```bash
gradle testDebugUnitTest assembleDebug
```

APK output:

```text
build/outputs/apk/debug/app-debug.apk
```

Local SDK/NDK/JDK installations, caches, model files, signing material and deployment configuration are kept outside the repository.

## Upstream projects and references

### Qualcomm AI Hub Apps / GenieX

Rin NPU Agent started from Qualcomm's `geniex_chat_android` application and continues to use GenieX as the local model runtime and model-management layer.

- https://github.com/qualcomm/ai-hub-apps
- https://github.com/qualcomm/geniex

### Model-To-NPU

The SDXL/QNN phone runtime follows and extends runtime work from Model-To-NPU:

- https://github.com/VitalikDen0/Model-To-NPU

This covers the Snapdragon 8 Elite QNN/HTP SDXL route, split UNet contexts, Lightning 8-step execution, runtime progress protocol, live-preview plumbing and the QNN LoRA path.

Model-To-NPU-derived runtime files retain the upstream PolyForm Noncommercial 1.0.0 license and Required Notice. See [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`LICENSES/`](LICENSES/).

### Model-side projects

The image path uses WAI Illustrious SDXL and ByteDance SDXL-Lightning. Precompiled packages are distributed through the model catalog and GitHub Releases.

## Licensing

- Qualcomm AI Hub Apps-derived code: BSD-3-Clause.
- Model-To-NPU-derived runtime components: PolyForm Noncommercial 1.0.0.
- Other dependencies and model packages follow their respective upstream licenses.

See [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`LICENSES/`](LICENSES/).
