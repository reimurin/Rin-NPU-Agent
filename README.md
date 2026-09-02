# Rin NPU Agent

[简体中文](README.zh-CN.md) | English

> Android ARM64 on-device AI Agent for Snapdragon devices, with local LLM/VLM chat, project-scoped file Agent capabilities, persistent projects/conversations, and an experimental WAI/SDXL NPU image-generation mode.

**Current public debug build:** `v1.5` · `versionCode 6` · package `com.geniex.demo` · **arm64-v8a only**

Rin NPU Agent is a personal/research-oriented Android AI workspace built around Qualcomm's on-device AI stack. The project started from Qualcomm's GenieX Android demo and has since been expanded into a project/conversation-oriented Agent application, with a second image-generation workspace designed around Snapdragon 8 Elite / SM8750 NPU deployment.

## Current status

### Chat / Agent mode

- Local LLM and VLM execution through GenieX.
- NPU preferred where a compatible QAIRT model is available; GPU/CPU remain available for compatible runtimes.
- Persistent **Project → Conversation → Message** hierarchy.
- Conversations are isolated by default.
- Cross-project / cross-conversation context can only be exposed to the model on an explicit user request.
- Per-project SAF workspace authorization for file tools.
- Agent tools include directory listing, file reading/writing, directory creation, and local PowerPoint generation.
- Model library uses runtime download/selection; **LLM/VLM model weights are not bundled in the APK**.

### Image-generation mode

The current UI/runtime integration targets the **WAI Illustrious SDXL + SDXL-Lightning 8-step** route on Snapdragon 8 Elite / SM8750.

Implemented in the app:

- Chat / Image mode switching.
- Independent positive-prompt and negative-prompt preset libraries.
- Positive presets support name + notes + full prompt.
- Negative presets are independent, reusable, and can be set as default/locked.
- Resolution presets with `1024×1024` as the default.
- Stable 8-step profile UI.
- Real pipeline progress parsing (`CLIP → UNet 1/8…8/8 → VAE → save`).
- Intermediate-preview plumbing through `preview_current.png` when the installed runtime supports it.
- Runtime/environment checks instead of fake “ready” states.
- Image-runtime memory handoff: the chat model can be released before SDXL generation to free RAM.

> [!IMPORTANT]
> **The SM8750 precompiled WAI QNN context package is not published in this repository yet.**
> The v1.5 APK contains the image-generation UI, installer/runtime framework, and phone-side runtime drivers, but actual NPU image generation still requires the compatible `sdxl_qnn` context package. Until that package is published/installed, the app will show a message similar to “NPU image model package not published yet”. This is expected and does not mean the APK installation itself failed.

## Debug APK

The GitHub Release for this repository contains the current ARM64 debug APK:

`Rin-NPU-Agent-v1.5-debug-upgrade.apk`

- ABI: `arm64-v8a`
- Package: `com.geniex.demo`
- Version: `1.5` (`versionCode 6`)
- Embedded GGUF/test model: **none**
- SHA-256: `f0439cacc80830baaf0ba7cc7d15c5cc494499d581fc0f779420a302d4e7a764`
- Debug signer SHA-256: `afeef7cf03c2bc3b411932df89330af1a6416f62e379096d72807305e3d9f801`

Because this is a debug build, it is intended for development/testing. In-place upgrade works only when the installed build uses the same signing certificate and a lower versionCode.

## UI design language: Rin Design System

The current interface follows an internal design language called **Rin Design System**, with **integrity** as a hard requirement: a new control is not considered finished if it visually falls back to an unrelated/default Android style.

Core rules:

- Unified rounded geometry rather than sharp default Android controls.
- Soft, low-contrast strokes and surface separation.
- Consistent card/input/button hierarchy across Chat, Drawer, Model Library and Image mode.
- Primary / secondary / ghost / destructive action hierarchy.
- Restrained typography levels for title, section, body and caption/status text.
- Spacing rhythm built around `8 / 12 / 16 / 24 dp`.
- Minimal “debug-tool” visual noise; low-level runtime information is visually secondary.
- New controls should reuse centralized colors/styles/drawables instead of raw hex values or system `<Button>` styling.

The visual direction is Material-inspired, but the goal is not to reproduce a stock Material screen. The emphasis is a softer, flatter, more continuous product language shared by all modes.

## Architecture

```text
Rin NPU Agent
├─ Chat mode
│  ├─ GenieX model manager
│  ├─ LLM / VLM runtime
│  ├─ Projects
│  │  └─ Conversations
│  └─ Project-scoped Agent workspace (SAF)
│
└─ Image mode
   ├─ Positive prompt library
   ├─ Negative prompt library
   ├─ Resolution / 8-step profile
   ├─ WAI/SDXL phone runtime driver
   ├─ QNN context/runtime checks
   └─ Real progress + optional intermediate preview
```

## Target platform

The primary development target is a **Snapdragon 8 Elite (SM8750)** Android phone. The APK itself is ARM64-only.

Build configuration currently uses:

- `minSdk 31`
- `targetSdk 34`
- `compileSdk 34`
- Java 17
- Android Gradle Plugin 8.13.0
- Kotlin 2.2.0
- NDK 27.3.13750724
- GenieX Android `0.3.5`

Other Snapdragon devices may work for chat models depending on the model/runtime package. Image-generation QNN contexts are chipset/runtime-specific and should not be assumed portable across SoCs.

## Build

Open the repository root in Android Studio, or use a compatible Gradle 8.13 installation with JDK 17 and Android SDK 34.

```bash
gradle assembleDebug
```

The generated APK is normally written to:

```text
build/outputs/apk/debug/app-debug.apk
```

`local.properties`, Android SDK/NDK/JDK installations, Gradle caches, model caches, build outputs, signing material, and personal deployment configuration are deliberately not committed.

## Projects referenced / upstream work

### Qualcomm AI Hub Apps / GenieX

This project started from Qualcomm's `geniex_chat_android` example in:

- https://github.com/qualcomm/ai-hub-apps
- GenieX SDK: https://github.com/qualcomm/geniex

The Qualcomm baseline provided the original Android GenieX integration, model-management structure, and pluggable local inference path. Large parts of the application architecture and UI have since been replaced or extended.

### Model-To-NPU

The SDXL-on-phone research and runtime integration references:

- https://github.com/VitalikDen0/Model-To-NPU

In particular, it informed the Snapdragon 8 Elite QNN/HTP SDXL route, split-UNet context approach, 8-step Lightning profile, stdout progress protocol, intermediate-preview mechanism, and LoRA context-slot direction.

The included Model-To-NPU-derived runtime driver files remain under **PolyForm Noncommercial License 1.0.0** and retain the upstream `LICENSE` / `NOTICE` / Required Notice lines. See [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`LICENSES/`](LICENSES/).

### Models / model-side projects

The intended image route references WAI Illustrious SDXL and ByteDance SDXL-Lightning. Model weights and precompiled QNN contexts are **not redistributed in this source repository** and remain subject to their own upstream licenses/terms.

## Third-party licensing

This repository contains code with different upstream licensing origins. Do not assume a single repository-wide license applies to every file.

- Qualcomm AI Hub Apps-derived code: BSD-3-Clause; see `LICENSES/QUALCOMM-AI-HUB-APPS-BSD-3-Clause.txt`.
- Model-To-NPU-derived runtime components: PolyForm Noncommercial 1.0.0; see `LICENSES/MODEL-TO-NPU-PolyForm-Noncommercial-1.0.0.txt` and its NOTICE.
- Other dependencies/models remain under their respective licenses.

See [`THIRD_PARTY.md`](THIRD_PARTY.md) for details.

## Roadmap

- Publish/install the SM8750 WAI SDXL QNN context package.
- Complete one-tap download → SHA-256 verification → installation for image contexts.
- Add a small local prompt-expansion model that is loaded on demand and immediately unloaded before SDXL generation.
- Automatic LoRA discovery by trigger word and QNN-compatible context-slot hot swap.
- Continue applying Rin Design System integrity rules to every new surface/control.

## Disclaimer

This is an independent personal/research project and is not an official Qualcomm, WAI, ByteDance, or Model-To-NPU product.
