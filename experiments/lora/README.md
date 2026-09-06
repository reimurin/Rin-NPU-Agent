# LoRA 1.6 实验记录

## 当前结果

`prototype.py` 和 `test_contract.py` 是标签、文件头与权重增量原型，41 项主机检查通过。`compile_dynamic_lora.py` 的动态 A/B 输入小图完成 7 组 CPU 数值检查和 SoC 69 / V79 编译。Android 侧新增的 21 项 Kotlin/JUnit 检查通过。手机 NPU 与完整 WAI LoRA 仍待验证。

`prepare_adapters.py`、`check_adapter_configs.py`、`compile_targeted_adapters.py` 保留预编译适配器试验。使用 QAIRT API 的 chipset:SM8750 才得到已核验的 V79 底模及三份适配器；早期仅设置 htp_socs 的命令曾退回 V68，这些错误目标产物没有打进 APK。enable_weights_updates、四维 alpha 和静态线性图的失败记录也保留，便于追溯而不是反复盲试。

`probe_broadcast_shape.py`、`probe_linear_update.py` 是失败用例，不是正式编译入口。所有脚本需要本地授权 QAIRT 2.48 和项目既有 Python。多份后续实验脚本使用项目固定路径；首次运行前核对输出目录，不在已有实验目录中覆盖执行。

## 测试包

参见 `docs/RELEASE_1.6.0-alpha.1.md`。默认入口测试动态权重；对照入口测试二进制分段。两个入口各执行 7 组真实 NPU 计算并比较数值，只有手机运行通过后才能判定该路径在目标设备有效。

动态路线后续需要在 WAI 中加入固定秩槽位、正确的层名/形状映射及适配器输入缓存。必须再检查无 LoRA 基线、多 LoRA 混合、text encoder 修改和峰值内存。不能由一个小图推断完整模型已兼容。
