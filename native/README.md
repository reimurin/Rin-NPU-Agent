# Native dependencies for 1.5.11

`jni/` contains the custom Rin QNN bridge and the NDK makefiles used by the tested build. `QNN_SAMPLEAPP_2.48.patch` records the FP16 I/O, diagnostics and checked tensor-file changes against Qualcomm QAIRT 2.48.0.260626 SampleApp. Obtain the matching SDK from Qualcomm under its terms, use a working copy, and apply the patch from that SDK root.

The makefiles expect `qnn-native-sdk-2.48/qairt/2.48.0.260626` beside the repository `native` directory. Use NDK 27.3.13750724 and Android API 31 or later. The project also requires the pinned QNN runtime and the relocatable Python 3.13 / NumPy / Pillow native payload. Their path map is `src/main/assets/sdxl_runtime/python_native_map.json`.

The source archive is not a self-contained SDK or model bundle. Vendor libraries, SDK headers, model weights, private signing material and local deployment configuration are excluded. The tested APK includes its required runtime payload; the model catalog supplies the separate model data.

For exact reproduction, preserve the QAIRT 2.48 DSP ELF bytes after Android packaging, run zipalign, and sign with a certificate you control. The distributed application uses the project's existing certificate for upgrade compatibility; its private key is not distributed. Source snapshot hashes are in `docs/source-snapshot-1.5.11.json`.
