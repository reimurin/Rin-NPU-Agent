namespace {

struct RinHtpPerfClient {
  qnn::tools::sample_app::QnnFunctionPointers qnnFunctionPointers{};
  void* backendHandle = nullptr;
  void* modelHandle = nullptr;
  QnnHtpDevice_PerfInfrastructure_t perfInfra = QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT;
  uint32_t powerConfigId = 0;
  uint32_t deviceId = 0;
  uint32_t coreId = 0;
  int mode = 0;
  bool active = false;
};

std::mutex gRinHtpPerfMutex;
RinHtpPerfClient gRinHtpPerf;

void rinHtpPerfResetLocked() {
  if (gRinHtpPerf.powerConfigId != 0 && gRinHtpPerf.perfInfra.destroyPowerConfigId) {
    const auto err = gRinHtpPerf.perfInfra.destroyPowerConfigId(gRinHtpPerf.powerConfigId);
    __android_log_print(err == QNN_SUCCESS ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                        kTag,
                        "HTP perf destroy id=%u err=%u",
                        gRinHtpPerf.powerConfigId,
                        static_cast<unsigned>(err));
  }
  if (gRinHtpPerf.backendHandle) {
    pal::dynamicloading::dlClose(gRinHtpPerf.backendHandle);
  }
  if (gRinHtpPerf.modelHandle) {
    pal::dynamicloading::dlClose(gRinHtpPerf.modelHandle);
  }
  gRinHtpPerf = RinHtpPerfClient{};
}

std::string rinHtpPerfApplyLocked(int requestedMode) {
  if (!gRinHtpPerf.active || gRinHtpPerf.powerConfigId == 0 ||
      !gRinHtpPerf.perfInfra.setPowerConfig) {
    return "inactive";
  }

  const int mode = requestedMode <= 1 ? 1 : 2;
  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto& cfg = dcvs.dcvsV3Config;
  cfg.contextId = gRinHtpPerf.powerConfigId;
  cfg.setDcvsEnable = 1;
  cfg.dcvsEnable = 1;
  cfg.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  cfg.setSleepLatency = 1;
  cfg.sleepLatency = 40;
  cfg.setSleepDisable = 1;
  cfg.sleepDisable = 1;
  cfg.setBusParams = 1;
  cfg.setCoreParams = 1;

  if (mode == 2) {
    cfg.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
    cfg.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    cfg.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
    cfg.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  } else {
    cfg.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
    cfg.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
    cfg.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
    cfg.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
    cfg.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
    cfg.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
  }

  const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, nullptr};
  auto err = gRinHtpPerf.perfInfra.setPowerConfig(gRinHtpPerf.powerConfigId, configs);
  bool fallback = false;
  if (err != QNN_SUCCESS) {
    // Some firmware accepts the common DCVS controls but rejects explicit corners.
    cfg.setBusParams = 0;
    cfg.setCoreParams = 0;
    err = gRinHtpPerf.perfInfra.setPowerConfig(gRinHtpPerf.powerConfigId, configs);
    fallback = true;
  }
  if (err != QNN_SUCCESS) {
    std::ostringstream out;
    out << "setPowerConfig failed err=" << static_cast<unsigned>(err);
    __android_log_print(ANDROID_LOG_WARN, kTag, "HTP perf %s", out.str().c_str());
    return out.str();
  }
  gRinHtpPerf.mode = mode;
  std::ostringstream out;
  out << "ok mode=" << (mode == 2 ? "boost" : "balanced")
      << " id=" << gRinHtpPerf.powerConfigId
      << " device=" << gRinHtpPerf.deviceId
      << " core=" << gRinHtpPerf.coreId
      << " fallback=" << (fallback ? 1 : 0);
  __android_log_print(ANDROID_LOG_INFO, kTag, "HTP perf %s", out.str().c_str());
  return out.str();
}

std::string rinHtpPerfBeginLocked(const std::string& backendPath, int mode) {
  rinHtpPerfResetLocked();
  if (backendPath.empty()) return "backend path empty";

  using DynamicStatus = qnn::tools::dynamicloadutil::StatusCode;
  const auto loadStatus = qnn::tools::dynamicloadutil::getQnnFunctionPointers(
      backendPath,
      "",
      &gRinHtpPerf.qnnFunctionPointers,
      &gRinHtpPerf.backendHandle,
      false,
      &gRinHtpPerf.modelHandle);
  if (loadStatus != DynamicStatus::SUCCESS) {
    rinHtpPerfResetLocked();
    return "getQnnFunctionPointers failed=" + std::to_string(static_cast<int>(loadStatus));
  }

  auto& qnn = gRinHtpPerf.qnnFunctionPointers.qnnInterface;
  if (!qnn.deviceGetInfrastructure) {
    rinHtpPerfResetLocked();
    return "deviceGetInfrastructure unsupported";
  }

  QnnDevice_Infrastructure_t infra = nullptr;
  const auto infraErr = qnn.deviceGetInfrastructure(&infra);
  if (infraErr != QNN_SUCCESS || !infra) {
    std::string result = "deviceGetInfrastructure failed=" + std::to_string(static_cast<unsigned>(infraErr));
    rinHtpPerfResetLocked();
    return result;
  }
  auto* htpInfra = reinterpret_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htpInfra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF ||
      !htpInfra->perfInfra.createPowerConfigId ||
      !htpInfra->perfInfra.destroyPowerConfigId ||
      !htpInfra->perfInfra.setPowerConfig) {
    rinHtpPerfResetLocked();
    return "HTP PerfInfrastructure incomplete";
  }
  gRinHtpPerf.perfInfra = htpInfra->perfInfra;

  uint32_t deviceId = 0;
  uint32_t coreId = 0;
  if (qnn.deviceGetPlatformInfo && qnn.deviceFreePlatformInfo) {
    const QnnDevice_PlatformInfo_t* platform = nullptr;
    const auto platformErr = qnn.deviceGetPlatformInfo(nullptr, &platform);
    if (platformErr == QNN_SUCCESS && platform &&
        platform->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1 &&
        platform->v1.numHwDevices > 0 && platform->v1.hwDevices) {
      const auto& hw = platform->v1.hwDevices[0];
      if (hw.version == QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1) {
        deviceId = hw.v1.deviceId;
        if (hw.v1.numCores > 0 && hw.v1.cores &&
            hw.v1.cores[0].version == QNN_DEVICE_CORE_INFO_VERSION_1) {
          coreId = hw.v1.cores[0].v1.coreId;
        }
      }
    }
    if (platform) qnn.deviceFreePlatformInfo(nullptr, platform);
  }

  uint32_t id = 0;
  const auto createErr = gRinHtpPerf.perfInfra.createPowerConfigId(deviceId, coreId, &id);
  if (createErr != QNN_SUCCESS || id == 0) {
    std::string result = "createPowerConfigId failed=" + std::to_string(static_cast<unsigned>(createErr));
    rinHtpPerfResetLocked();
    return result;
  }
  gRinHtpPerf.powerConfigId = id;
  gRinHtpPerf.deviceId = deviceId;
  gRinHtpPerf.coreId = coreId;
  gRinHtpPerf.active = true;
  return rinHtpPerfApplyLocked(mode);
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_geniex_demo_image_QnnInProcessNative_beginHtpPerformance(
    JNIEnv* env, jobject /* thiz */, jstring backendPathJ, jint mode) {
  std::lock_guard<std::mutex> lock(gRinHtpPerfMutex);
  const std::string backendPath = JUtfString(env, backendPathJ).str();
  const std::string result = rinHtpPerfBeginLocked(backendPath, static_cast<int>(mode));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_geniex_demo_image_QnnInProcessNative_setHtpPerformanceMode(
    JNIEnv* env, jobject /* thiz */, jint mode) {
  std::lock_guard<std::mutex> lock(gRinHtpPerfMutex);
  const std::string result = rinHtpPerfApplyLocked(static_cast<int>(mode));
  return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_geniex_demo_image_QnnInProcessNative_endHtpPerformance(
    JNIEnv* env, jobject /* thiz */) {
  std::lock_guard<std::mutex> lock(gRinHtpPerfMutex);
  const uint32_t oldId = gRinHtpPerf.powerConfigId;
  rinHtpPerfResetLocked();
  const std::string result = "released id=" + std::to_string(oldId);
  return env->NewStringUTF(result.c_str());
}
