# 1.6 alpha 构建依赖

本分支沿用 1.5.11 的可用 QNN 核心库、Python native runtime 和历史签名。`qnn-native-sdk-2.48/`、`src/main/jniLibs/`、`native/obj/`、`native/libs/` 都是本地依赖或构建输出，不提交仓库。

首次准备时，把 1.5.11 已验证的 `toolchain/qnn-native-sdk-2.48` 复制到本工作树根目录，再应用 `native/sampleapp_lora_sequence.patch`。该补丁依赖稳定版本已完成的 FP16/I/O 修复，不直接用于未经修补的原始 SDK。不要把新补丁写回稳定版本的 SDK 工作副本。将稳定 APK 对应的 jniLibs 复制到本分支的 `src/main/jniLibs/arm64-v8a/`。

构建入口：

```text
python scripts/build_lora_alpha.py --toolchain H:\Rin_Android_NPU_Agent\toolchain --out <新的独立构建目录>
```

脚本先构建 JNI，再运行 testDebugUnitTest 和 assembleDebug，最后使用现有工具链恢复 DSP 原字节、zipalign、历史证书签名。DSP 后处理脚本仍来自本机稳定工具链，外部复现需按 1.5.11 构建文档准备这份依赖。不要发布未经后处理的 Gradle 原始 APK。

APP ID 与正式版不同，只有实验室为启动入口。APK 自测素材都是本项目生成的小型数值实验数据，不含 WAI 权重。被跟踪的素材 JSON 记录各文件 SHA；更新素材时必须同时更新校验记录并重新构建。
