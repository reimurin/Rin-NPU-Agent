# LoRA 1.6 原型

这组实验独立于正式 APP，不会加载 WAI 或生成图片。`prototype.py` 解析 LoRA 与普通词权重，检查 Safetensors 结构，演示数值增量；尚未接入手机 UI、CLIP 权重或 QNN adapter 执行。

运行环境使用项目既有 Python 3.10 和 numpy、onnx、onnxruntime、safetensors；QAIRT 实验使用本地授权的 2.48 SDK。实验输出应放在 Git 工作树之外的新临时目录，不提交模型二进制或 SDK。

```text
python test_contract.py --out-dir <new-test-output-directory>
python smoke_qairt_lora.py --sdk-root <local-qairt-2.48> --out-dir <new-smoke-output-directory>
```

首轮主机检查为 41/41；小图已生成可更新的 SM8750 context，详见 `offline-proof.json`。该记录不包含手机 NPU 或 adapter 热切换验证。下一步先补齐小图 adapter binary 生成与应用，再映射完整 WAI。

原型使用保守权重范围：LoRA -2 至 2，单层普通词 0 至 2；这只是解析测试范围，不代表已测出的最佳生图范围。文件头可读者一律标记为待 QNN 模板适配。LoRA 原文件目录由 runtime base 派生为 `Lora/`，适配缓存为 `.rin_lora/cache/`，两者分开管理。
