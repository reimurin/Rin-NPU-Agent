#include <jni.h>
#include <android/log.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnSampleApp.hpp"
#include "QnnSampleAppUtils.hpp"

namespace {

constexpr const char* kTag = "RinQnnBridge";
std::mutex gRunMutex;
std::once_flag gLogInitOnce;
bool gLogReady = false;

std::mutex gNativeLogMutex;
std::string gNativeLog;

void clearNativeLog() {
  std::lock_guard<std::mutex> lock(gNativeLogMutex);
  gNativeLog.clear();
}

std::string nativeLogTail() {
  std::lock_guard<std::mutex> lock(gNativeLogMutex);
  constexpr size_t kMaxReturn = 12000;
  if (gNativeLog.size() <= kMaxReturn) return gNativeLog;
  return gNativeLog.substr(gNativeLog.size() - kMaxReturn);
}

const char* levelName(QnnLog_Level_t level) {
  switch (level) {
    case QNN_LOG_LEVEL_ERROR: return "ERROR";
    case QNN_LOG_LEVEL_WARN: return "WARN";
    case QNN_LOG_LEVEL_INFO: return "INFO";
    case QNN_LOG_LEVEL_DEBUG: return "DEBUG";
    case QNN_LOG_LEVEL_VERBOSE: return "VERBOSE";
    default: return "UNKNOWN";
  }
}

void rinQnnLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp) {
  char message[4096];
  va_list copy;
  va_copy(copy, argp);
  std::vsnprintf(message, sizeof(message), fmt ? fmt : "", copy);
  va_end(copy);
  char line[4600];
  std::snprintf(line, sizeof(line), "%.3fms [%s] %s", static_cast<double>(timestamp) / 1000000.0, levelName(level), message);
  __android_log_print(level == QNN_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, kTag, "%s", line);
  std::lock_guard<std::mutex> lock(gNativeLogMutex);
  gNativeLog.append(line);
  gNativeLog.push_back('\n');
  constexpr size_t kMaxBuffer = 64000;
  if (gNativeLog.size() > kMaxBuffer) gNativeLog.erase(0, gNativeLog.size() - kMaxBuffer);
}

class JUtfString {
 public:
  JUtfString(JNIEnv* env, jstring value) : env_(env), value_(value) {
    if (value_) chars_ = env_->GetStringUTFChars(value_, nullptr);
  }
  ~JUtfString() {
    if (chars_) env_->ReleaseStringUTFChars(value_, chars_);
  }
  std::string str() const { return chars_ ? std::string(chars_) : std::string(); }

 private:
  JNIEnv* env_;
  jstring value_;
  const char* chars_ = nullptr;
};

std::string jsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          static const char* hex = "0123456789abcdef";
          out << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

jstring makeResult(JNIEnv* env,
                   bool ok,
                   const std::string& stage,
                   const std::string& detail,
                   double elapsedMs,
                   const std::string& backendBuild,
                   const std::string& nativeLog) {
  std::ostringstream out;
  out << "{\"ok\":" << (ok ? "true" : "false")
      << ",\"stage\":\"" << jsonEscape(stage) << "\""
      << ",\"detail\":\"" << jsonEscape(detail) << "\""
      << ",\"elapsed_ms\":" << elapsedMs
      << ",\"backend_build\":\"" << jsonEscape(backendBuild) << "\""
      << ",\"native_log\":\"" << jsonEscape(nativeLog) << "\"}";
  return env->NewStringUTF(out.str().c_str());
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_geniex_demo_image_QnnInProcessNative_runContext(
    JNIEnv* env,
    jobject /* thiz */,
    jstring backendPathJ,
    jstring systemLibraryPathJ,
    jstring contextPathJ,
    jstring inputListPathJ,
    jstring outputDirJ,
    jboolean nativeInput,
    jboolean nativeOutput) {
  std::lock_guard<std::mutex> lock(gRunMutex);
  const auto started = std::chrono::steady_clock::now();
  clearNativeLog();

  const std::string backendPath = JUtfString(env, backendPathJ).str();
  const std::string systemLibraryPath = JUtfString(env, systemLibraryPathJ).str();
  const std::string contextPath = JUtfString(env, contextPathJ).str();
  const std::string inputListPath = JUtfString(env, inputListPathJ).str();
  const std::string outputDir = JUtfString(env, outputDirJ).str();

  auto elapsedMs = [&]() -> double {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started)
        .count();
  };

  if (backendPath.empty() || systemLibraryPath.empty() || contextPath.empty() ||
      inputListPath.empty() || outputDir.empty()) {
    return makeResult(env, false, "arguments", "missing required path", elapsedMs(), "", nativeLogTail());
  }

  std::call_once(gLogInitOnce, []() {
    gLogReady = qnn::log::initializeLogging(rinQnnLogCallback, QNN_LOG_LEVEL_VERBOSE);
  });
  if (!gLogReady) {
    return makeResult(env, false, "logging", "QNN logging initialization failed", elapsedMs(), "", nativeLogTail());
  }

  using qnn::tools::dynamicloadutil::StatusCode;
  using AppStatus = qnn::tools::sample_app::StatusCode;

  qnn::tools::sample_app::QnnFunctionPointers qnnFunctionPointers{};
  void* backendHandle = nullptr;
  void* modelHandle = nullptr;
  std::unique_ptr<qnn::tools::sample_app::QnnSampleApp> app;
  bool deviceCreated = false;
  bool contextCreated = false;
  AppStatus devicePropertySupportStatus = AppStatus::FAILURE;
  std::string backendBuild;
  std::string stage = "load_backend";
  std::string detail;
  std::string operationLog;
  bool ok = false;

  __android_log_print(ANDROID_LOG_INFO, kTag, "runContext ctx=%s", contextPath.c_str());

  auto loadStatus = qnn::tools::dynamicloadutil::getQnnFunctionPointers(
      backendPath,
      "",
      &qnnFunctionPointers,
      &backendHandle,
      false,
      &modelHandle);
  if (loadStatus != StatusCode::SUCCESS) {
    detail = "getQnnFunctionPointers failed=" + std::to_string(static_cast<int>(loadStatus));
    goto cleanup;
  }

  stage = "load_system";
  loadStatus = qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
      systemLibraryPath, &qnnFunctionPointers);
  if (loadStatus != StatusCode::SUCCESS) {
    detail = "getQnnSystemFunctionPointers failed=" + std::to_string(static_cast<int>(loadStatus));
    goto cleanup;
  }

  stage = "construct";
  app = std::make_unique<qnn::tools::sample_app::QnnSampleApp>(
      qnnFunctionPointers,
      inputListPath,
      "",
      backendHandle,
      outputDir,
      false,
      nativeOutput == JNI_TRUE ? qnn::tools::iotensor::OutputDataType::NATIVE_ONLY
                               : qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
      nativeInput == JNI_TRUE ? qnn::tools::iotensor::InputDataType::NATIVE
                              : qnn::tools::iotensor::InputDataType::FLOAT,
      qnn::tools::sample_app::ProfilingLevel::OFF,
      true,
      contextPath,
      "",
      1,
      false,
      "");
  backendBuild = app->getBackendBuildId();

  stage = "initialize";
  if (app->initialize() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::initialize failed";
    goto cleanup;
  }

  stage = "initialize_backend";
  if (app->initializeBackend() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::initializeBackend failed";
    goto cleanup;
  }

  stage = "device_property";
  devicePropertySupportStatus = app->isDevicePropertySupported();
  if (devicePropertySupportStatus != AppStatus::FAILURE) {
    stage = "create_device";
    deviceCreated = (app->createDevice() == AppStatus::SUCCESS);
    if (!deviceCreated) {
      detail = "QnnSampleApp::createDevice failed";
      goto cleanup;
    }
  }

  stage = "profiling";
  if (app->initializeProfiling() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::initializeProfiling failed";
    goto cleanup;
  }

  stage = "op_packages";
  if (app->registerOpPackages() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::registerOpPackages failed";
    goto cleanup;
  }

  stage = "create_from_binary";
  contextCreated = (app->createFromBinary() == AppStatus::SUCCESS);
  if (!contextCreated) {
    detail = "QnnSampleApp::createFromBinary failed";
    goto cleanup;
  }

  if (app->isFinalizeDeserializedGraphSupported() == AppStatus::SUCCESS) {
    stage = "finalize_graphs";
    if (app->finalizeGraphs() != AppStatus::SUCCESS) {
      detail = "QnnSampleApp::finalizeGraphs failed";
      goto cleanup;
    }
  }

  stage = "execute_graphs";
  if (app->executeGraphs() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::executeGraphs failed: " + app->getLastExecutionDetail();
    goto cleanup;
  }

  stage = "complete";
  ok = true;

cleanup:
  operationLog = nativeLogTail();  // Preserve the failure before cleanup floods the log.
  if (app) {
    if (contextCreated) {
      app->freeContext();
      contextCreated = false;
    }
    if (deviceCreated && devicePropertySupportStatus != AppStatus::FAILURE) {
      app->freeDevice();
      deviceCreated = false;
    }
    app->terminateBackend();
  }
  if (backendHandle) {
    pal::dynamicloading::dlClose(backendHandle);
    backendHandle = nullptr;
  }
  if (modelHandle) {
    pal::dynamicloading::dlClose(modelHandle);
    modelHandle = nullptr;
  }

  __android_log_print(ok ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR,
                      kTag,
                      "result ok=%d stage=%s detail=%s elapsed=%.1fms",
                      ok ? 1 : 0,
                      stage.c_str(),
                      detail.c_str(),
                      elapsedMs());
  return makeResult(env, ok, stage, detail, elapsedMs(), backendBuild, operationLog);
}

// Isolated 1.6 LoRA sequence self-test. The normal generation entry is unchanged.
extern "C" JNIEXPORT jstring JNICALL
Java_com_geniex_demo_image_QnnInProcessNative_runLoraSequence(
    JNIEnv* env,
    jobject /* thiz */,
    jstring backendPathJ,
    jstring systemLibraryPathJ,
    jstring contextPathJ,
    jstring inputListPathJ,
    jstring outputDirJ,
    jboolean nativeInput,
    jboolean nativeOutput,
    jstring testRootJ,
    jboolean applyBinaryAdapters) {
  std::lock_guard<std::mutex> lock(gRunMutex);
  const auto started = std::chrono::steady_clock::now();
  clearNativeLog();

  const std::string backendPath = JUtfString(env, backendPathJ).str();
  const std::string systemLibraryPath = JUtfString(env, systemLibraryPathJ).str();
  const std::string contextPath = JUtfString(env, contextPathJ).str();
  const std::string inputListPath = JUtfString(env, inputListPathJ).str();
  const std::string outputDir = JUtfString(env, outputDirJ).str();
  const std::string testRoot = JUtfString(env, testRootJ).str();

  auto elapsedMs = [&]() -> double {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started)
        .count();
  };

  if (backendPath.empty() || systemLibraryPath.empty() || contextPath.empty() ||
      inputListPath.empty() || outputDir.empty()) {
    return makeResult(env, false, "arguments", "missing required path", elapsedMs(), "", nativeLogTail());
  }

  std::call_once(gLogInitOnce, []() {
    gLogReady = qnn::log::initializeLogging(rinQnnLogCallback, QNN_LOG_LEVEL_VERBOSE);
  });
  if (!gLogReady) {
    return makeResult(env, false, "logging", "QNN logging initialization failed", elapsedMs(), "", nativeLogTail());
  }

  using qnn::tools::dynamicloadutil::StatusCode;
  using AppStatus = qnn::tools::sample_app::StatusCode;

  qnn::tools::sample_app::QnnFunctionPointers qnnFunctionPointers{};
  void* backendHandle = nullptr;
  void* modelHandle = nullptr;
  std::unique_ptr<qnn::tools::sample_app::QnnSampleApp> app;
  bool deviceCreated = false;
  bool contextCreated = false;
  AppStatus devicePropertySupportStatus = AppStatus::FAILURE;
  std::string backendBuild;
  std::string stage = "load_backend";
  std::string detail;
  std::string operationLog;
  bool ok = false;

  __android_log_print(ANDROID_LOG_INFO, kTag, "runContext ctx=%s", contextPath.c_str());

  auto loadStatus = qnn::tools::dynamicloadutil::getQnnFunctionPointers(
      backendPath,
      "",
      &qnnFunctionPointers,
      &backendHandle,
      false,
      &modelHandle);
  if (loadStatus != StatusCode::SUCCESS) {
    detail = "getQnnFunctionPointers failed=" + std::to_string(static_cast<int>(loadStatus));
    goto cleanup;
  }

  stage = "load_system";
  loadStatus = qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
      systemLibraryPath, &qnnFunctionPointers);
  if (loadStatus != StatusCode::SUCCESS) {
    detail = "getQnnSystemFunctionPointers failed=" + std::to_string(static_cast<int>(loadStatus));
    goto cleanup;
  }

  stage = "construct";
  app = std::make_unique<qnn::tools::sample_app::QnnSampleApp>(
      qnnFunctionPointers,
      inputListPath,
      "",
      backendHandle,
      outputDir,
      false,
      nativeOutput == JNI_TRUE ? qnn::tools::iotensor::OutputDataType::NATIVE_ONLY
                               : qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
      nativeInput == JNI_TRUE ? qnn::tools::iotensor::InputDataType::NATIVE
                              : qnn::tools::iotensor::InputDataType::FLOAT,
      qnn::tools::sample_app::ProfilingLevel::OFF,
      true,
      contextPath,
      "",
      1,
      false,
      "");
  backendBuild = app->getBackendBuildId();

  stage = "initialize";
  if (app->initialize() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::initialize failed";
    goto cleanup;
  }

  stage = "initialize_backend";
  if (app->initializeBackend() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::initializeBackend failed";
    goto cleanup;
  }

  stage = "device_property";
  devicePropertySupportStatus = app->isDevicePropertySupported();
  if (devicePropertySupportStatus != AppStatus::FAILURE) {
    stage = "create_device";
    deviceCreated = (app->createDevice() == AppStatus::SUCCESS);
    if (!deviceCreated) {
      detail = "QnnSampleApp::createDevice failed";
      goto cleanup;
    }
  }

  stage = "profiling";
  if (app->initializeProfiling() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::initializeProfiling failed";
    goto cleanup;
  }

  stage = "op_packages";
  if (app->registerOpPackages() != AppStatus::SUCCESS) {
    detail = "QnnSampleApp::registerOpPackages failed";
    goto cleanup;
  }

  stage = "create_from_binary";
  contextCreated = (app->createFromBinary() == AppStatus::SUCCESS);
  if (!contextCreated) {
    detail = "QnnSampleApp::createFromBinary failed";
    goto cleanup;
  }

  if (app->isFinalizeDeserializedGraphSupported() == AppStatus::SUCCESS) {
    stage = "finalize_graphs";
    if (app->finalizeGraphs() != AppStatus::SUCCESS) {
      detail = "QnnSampleApp::finalizeGraphs failed";
      goto cleanup;
    }
  }

  {
    const char* cases[] = {"base0", "original08", "original11", "changed08", "restored08", "restored0", "zero11"};
    const char* adapters[] = {"", "tiny_original.bin", "", "tiny_changed.bin", "tiny_original.bin", "", "tiny_zero.bin"};
    for (size_t i = 0; i < 7; ++i) {
      stage = std::string("lora_io_") + cases[i];
      if (app->rinSetRunIO(testRoot + "/" + cases[i] + "/inputs.txt", outputDir + "/" + cases[i]) != AppStatus::SUCCESS) {
        detail = app->getLastExecutionDetail(); goto cleanup;
      }
      if (*adapters[i] && applyBinaryAdapters == JNI_TRUE) {
        stage = std::string("lora_apply_") + cases[i];
        if (app->rinApplyAdapter(testRoot + "/assets/" + adapters[i]) != AppStatus::SUCCESS) {
          detail = app->getLastExecutionDetail(); goto cleanup;
        }
      }
      stage = std::string("lora_execute_") + cases[i];
      if (app->executeGraphs(false) != AppStatus::SUCCESS) {
        detail = app->getLastExecutionDetail(); goto cleanup;
      }
    }
  }

  stage = "complete";
  ok = true;

cleanup:
  operationLog = nativeLogTail();  // Preserve the failure before cleanup floods the log.
  if (app) {
    app->rinReleaseGraphMetadata();
    if (contextCreated) {
      app->freeContext();
      contextCreated = false;
    }
    if (deviceCreated && devicePropertySupportStatus != AppStatus::FAILURE) {
      app->freeDevice();
      deviceCreated = false;
    }
    app->terminateBackend();
    app.reset();
  }
  if (backendHandle) {
    pal::dynamicloading::dlClose(backendHandle);
    backendHandle = nullptr;
  }
  if (modelHandle) {
    pal::dynamicloading::dlClose(modelHandle);
    modelHandle = nullptr;
  }

  __android_log_print(ok ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR,
                      kTag,
                      "result ok=%d stage=%s detail=%s elapsed=%.1fms",
                      ok ? 1 : 0,
                      stage.c_str(),
                      detail.c_str(),
                      elapsedMs());
  return makeResult(env, ok, stage, detail, elapsedMs(), backendBuild, operationLog);
}
