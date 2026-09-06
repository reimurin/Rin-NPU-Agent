# Rin NPU Agent 1.6.0-alpha.1：LoRA 实验室

这是用于确定 LoRA 执行路径的独立测试包，不是已经完成 WAI LoRA 功能的正式 1.6.0。安装后显示“Rin LoRA 1.6 测试”，与 1.5.11 并列保留。无需卸载正式版，也无需重新下载 WAI 或 CLIP-G 修复包。

## 手机测试

打开测试 APP，授予文件访问权限。暂停正式版的生图任务，然后点击“开始动态 LoRA NPU 自测”。自测使用内置小模型，在同一 context 中检查强度 0、0.8、1.1、更换权重、恢复和归零的结果，不生成图片，不修改已有模型。结束后点击“分享自测报告”。

“对照测试：预编译适配器”使用另一组小图检查实际 contextApplyBinarySection 调用。两条路线的结果分别保存，分享按钮发送最近一次报告。目录为 `Download/sdxl_qnn/.rin_diagnostics/`，文件包括 `lora_selftest_latest.json`、`lora_selftest_dynamic_weights.json`、`lora_selftest_binary_sections.json`。

## 本版已包含

LoRA 原文件目录为 `Download/sdxl_qnn/Lora/`。页面能后台检查 `.safetensors` 结构，显示不完整、类型不支持等原因；结构可读的文件标记为“等待 WAI 适配”，不会假装已能生成。

直接标签支持 `<lora:名称:0.8>`、独立的多 LoRA 权重、默认 1.0、零强度与冲突标签检查。当前这里只做语法/文件入口，尚未把任意用户 LoRA 接入 WAI，也未完成普通词 embedding 加权或实时预览。

## 构建与验证

版本为 1.6.0-alpha.1 / code 18，applicationId 为 `com.geniex.demo.loratest`。21 项 Kotlin/JUnit 检查通过，动态权重小图 7 组 CPU 数值检查通过，两条路线的模型均已确认为 SM8750 / V79。APK 的 NDK/Kotlin 构建、49 项内置素材校验、9 个 QNN 核心库比对与历史签名检查通过。发布前没有手机 NPU 实测，不将离线编译等同于实机成功。

APK bytes：128,698,550
SHA-256：938c899389052acbe8e38866b4c081b7add5006cd4a43be6ceaee676d1626a74

稳定 Latest 继续为 1.5.11。本 alpha 不重传或覆盖模型包。
