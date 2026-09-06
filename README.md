> This branch builds 1.6.0-alpha.2 as an in-place Rin NPU Agent update. Startup and preset management are repaired; full WAI LoRA and additional native resolutions remain pending. See [alpha.2 release notes](docs/RELEASE_1.6.0-alpha.2.md).

# Rin NPU Agent

[简体中文](README.zh-CN.md)

Android ARM64 on-device AI workspace for Snapdragon NPU devices. Rin NPU Agent combines local LLM/VLM chat, project-scoped Agent tools, persistent projects/conversations, and a WAI/SDXL image-generation workspace in one application.

**Current stable version:** `1.5.11` · package `com.geniex.demo` · `arm64-v8a`

## Features

### Chat & Agent

- Local LLM and VLM inference through Qualcomm GenieX.
- NPU-first model selection where QAIRT packages are available, with compatible GPU/CPU runtimes as alternatives.
- Persistent **Project → Conversation → Message** hierarchy.
- Project-scoped SAF workspaces for Agent file operations.
- Cross-project and cross-conversation context retrieval on explicit request.
- Local file tools and local PowerPoint generation.
- In-app model library and model downloading; model weights are kept outside the APK.

### Image generation

The image workspace uses **WAI Illustrious SDXL + SDXL-Lightning 8-step** on Qualcomm HTP/QNN.

- Chat / Image mode switching.
- Independent positive and negative prompt libraries.
- Named positive presets with notes.
- Reusable/default/locked negative presets.
- Resolution presets with `1024×1024` as the default.
- Real pipeline progress: `CLIP → UNet 1/8 … 8/8 → VAE → save`.
- Intermediate preview through `preview_current.png` when the installed runtime provides preview decoding.
- Runtime readiness checks and memory handoff between the chat model and SDXL.
- Device-specific precompiled package installer.

The SM8750 runtime is published. The project user confirmed successful on-device generation with 1.5.11 on Honor Magic 7 Pro. See the [stable release](https://github.com/reimurin/Rin-NPU-Agent/releases/tag/v1.5.11) and [release notes](docs/RELEASE_1.5.11.md).

## Model package catalog

The canonical catalog is [`models/index.json`](models/index.json). Android clients fetch the fixed URL:

```text
https://raw.githubusercontent.com/reimurin/Rin-NPU-Agent/main/models/index.json
```

Each package entry records compatible chipset, ABI, runtime, resolutions, archive checksum and GitHub Release assets. The app matches the current device and selects the best published package.

Large QNN context bundles are stored as **GitHub Release assets**, not in Git history. GitHub requires each Release asset to stay below 2 GiB, so large archives are split into ordered parts. The installer downloads each part, verifies SHA-256, verifies the archive through a concatenated stream and extracts it without writing an intermediate ZIP. CLIP-G repair is downloaded separately.

Adding another Snapdragon generation or QNN profile only requires publishing another package and updating `models/index.json`; the APK keeps the same catalog endpoint.

## Versioning

Rin NPU Agent uses semantic versioning from this development line onward:

- `1.5.11`, `1.5.12`, ... for fixes and incremental runtime/UI improvements.
- `1.6.0`, `1.7.0`, ... for larger feature milestones.
- `2.0.0` for a future breaking application/runtime generation.

Android `versionCode` remains monotonically increasing for in-place upgrades.

## Rin Design System

The UI follows **Rin Design System**, with **integrity** as a project-wide design rule: every new control inherits the same visual language instead of falling back to an unrelated default Android style.

- Unified rounded geometry.
- Soft, low-contrast strokes and surface separation.
- Shared card, input, button, list and dialog hierarchy.
- Primary / secondary / ghost / destructive action levels.
- Consistent title / section / body / caption typography.
- `8 / 12 / 16 / 24 dp` spacing rhythm.
- Flat, restrained runtime/status presentation.
- Centralized colors, styles and drawables.

The direction is Material-inspired, with a softer and more continuous product language across Chat, Drawer, Model Library and Image mode.

## Architecture

```text
Rin NPU Agent
├─ Chat mode
│  ├─ GenieX model manager
│  ├─ LLM / VLM runtime
│  ├─ Projects → Conversations
│  └─ Project-scoped Agent workspace
│
└─ Image mode
   ├─ Positive prompt library
   ├─ Negative prompt library
   ├─ Resolution / 8-step profile
   ├─ GitHub model catalog
   ├─ Device-specific QNN package installer
   ├─ WAI/SDXL phone runtime driver
   └─ Real progress + intermediate preview
```

## Target platform

Primary development target: **Snapdragon 8 Elite / SM8750** Android devices.

- `minSdk 31`
- `targetSdk 34`
- `compileSdk 34`
- Java 17
- Android Gradle Plugin 8.13.0
- Kotlin 2.2.0
- NDK 27.3.13750724
- GenieX Android 0.3.5

## Build

First prepare the [native prerequisites](native/README.md). This source snapshot excludes SDK libraries and is not self-contained. With those prerequisites, use JDK 17, Android SDK 34 and Gradle 8.13:

```bash
gradle assembleDebug
```

APK output:

```text
build/outputs/apk/debug/app-debug.apk
```

Local SDK/NDK/JDK installations, caches, model files, build outputs, signing material and deployment configuration stay outside the repository.

## Upstream projects and references

### Qualcomm AI Hub Apps / GenieX

Rin NPU Agent started from Qualcomm's `geniex_chat_android` application and continues to use GenieX as the local model runtime/model-management layer.

- https://github.com/qualcomm/ai-hub-apps
- https://github.com/qualcomm/geniex

### Model-To-NPU

The SDXL/QNN phone runtime follows and extends ideas and runtime components from Model-To-NPU:

- https://github.com/VitalikDen0/Model-To-NPU

This includes the Snapdragon 8 Elite QNN/HTP SDXL route, split-UNet context layout, Lightning 8-step path, runtime progress protocol, intermediate-preview plumbing and QNN LoRA context-slot direction.

Model-To-NPU-derived runtime files retain the upstream PolyForm Noncommercial 1.0.0 license and Required Notice. See [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`LICENSES/`](LICENSES/).

### Model-side projects

The image path uses WAI Illustrious SDXL and ByteDance SDXL-Lightning. Precompiled packages are distributed through the model catalog and GitHub Releases.

## Licensing

- Qualcomm AI Hub Apps-derived code: BSD-3-Clause.
- Model-To-NPU-derived runtime components: PolyForm Noncommercial 1.0.0.
- Other dependencies and model packages follow their respective upstream licenses.

See [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`LICENSES/`](LICENSES/).

## Roadmap

- Keep 1.5.11 as the verified stable generation baseline.
- Add more resolution buckets and Snapdragon targets.
- Add on-demand local prompt expansion followed by immediate LLM unload before SDXL generation.
- Add trigger-word-based LoRA discovery and QNN context-slot switching.
- Continue extending Rin Design System across every new control and screen.

See the [1.6 plan](docs/PLAN_1.6.md) for LoRA import, preview and acceptance stages.
