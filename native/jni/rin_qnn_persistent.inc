namespace {

using RinPersistentAppStatus = qnn::tools::sample_app::StatusCode;

struct RinPersistentContext {
  qnn::tools::sample_app::QnnFunctionPointers qnnFunctionPointers{};
  void* backendHandle = nullptr;
  void* modelHandle = nullptr;
  std::unique_ptr<qnn::tools::sample_app::QnnSampleApp> app;
  bool deviceCreated = false;
  bool contextCreated = false;
  RinPersistentAppStatus devicePropertySupportStatus = RinPersistentAppStatus::FAILURE;
  std::string backendBuild;
};

std::unordered_map<std::string, std::unique_ptr<RinPersistentContext>> gRinPersistentContexts;
constexpr size_t kRinPersistentContextLimit = 8;

std::string rinPersistentKey(const std::string& backendPath,
                             const std::string& systemLibraryPath,
                             const std::string& contextPath,
                             bool nativeInput,
                             bool nativeOutput) {
  return backendPath + "\n" + systemLibraryPath + "\n" + contextPath + "\n" +
         (nativeInput ? "native-in" : "float-in") + "\n" +
         (nativeOutput ? "native-out" : "float-out");
}

void rinCleanupPersistentContext(RinPersistentContext& session) {
  if (session.app) {
    session.app->rinReleaseGraphMetadata();
    if (session.contextCreated) {
      session.app->freeContext();
      session.contextCreated = false;
    }
    if (session.deviceCreated &&
        session.devicePropertySupportStatus != RinPersistentAppStatus::FAILURE) {
      session.app->freeDevice();
      session.deviceCreated = false;
    }
    session.app->terminateBackend();
    session.app.reset();
  }
  if (session.backendHandle) {
    pal::dynamicloading::dlClose(session.backendHandle);
    session.backendHandle = nullptr;
  }
  if (session.modelHandle) {
    pal::dynamicloading::dlClose(session.modelHandle);
    session.modelHandle = nullptr;
  }
}

void rinReleaseAllPersistentContexts() {
  for (auto& item : gRinPersistentContexts) {
    if (item.second) rinCleanupPersistentContext(*item.second);
  }
  gRinPersistentContexts.clear();
}

std::unique_ptr<RinPersistentContext> rinCreatePersistentContext(
    const std::string& backendPath,
    const std::string& systemLibraryPath,
    const std::string& contextPath,
    const std::string& inputListPath,
    const std::string& outputDir,
    bool nativeInput,
    bool nativeOutput,
    std::string& stage,
    std::string& detail) {
  using DynamicStatus = qnn::tools::dynamicloadutil::StatusCode;
  auto session = std::make_unique<RinPersistentContext>();

  stage = "persistent_load_backend";
  auto loadStatus = qnn::tools::dynamicloadutil::getQnnFunctionPointers(
      backendPath,
      "",
      &session->qnnFunctionPointers,
      &session->backendHandle,
      false,
      &session->modelHandle);
  if (loadStatus != DynamicStatus::SUCCESS) {
    detail = "getQnnFunctionPointers failed=" +
             std::to_string(static_cast<int>(loadStatus));
    rinCleanupPersistentContext(*session);
    return nullptr;
  }

  stage = "persistent_load_system";
  loadStatus = qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
      systemLibraryPath, &session->qnnFunctionPointers);
  if (loadStatus != DynamicStatus::SUCCESS) {
    detail = "getQnnSystemFunctionPointers failed=" +
             std::to_string(static_cast<int>(loadStatus));
    rinCleanupPersistentContext(*session);
    return nullptr;
  }

  stage = "persistent_construct";
  session->app = std::make_unique<qnn::tools::sample_app::QnnSampleApp>(
      session->qnnFunctionPointers,
      inputListPath,
      "",
      session->backendHandle,
      outputDir,
      false,
      nativeOutput ? qnn::tools::iotensor::OutputDataType::NATIVE_ONLY
                   : qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
      nativeInput ? qnn::tools::iotensor::InputDataType::NATIVE
                  : qnn::tools::iotensor::InputDataType::FLOAT,
      qnn::tools::sample_app::ProfilingLevel::OFF,
      true,
      contextPath,
      "",
      1,
      false,
      "");
  session->backendBuild = session->app->getBackendBuildId();

  stage = "persistent_initialize";
  if (session->app->initialize() != RinPersistentAppStatus::SUCCESS) {
    detail = "QnnSampleApp::initialize failed";
    rinCleanupPersistentContext(*session);
    return nullptr;
  }

  stage = "persistent_initialize_backend";
  if (session->app->initializeBackend() != RinPersistentAppStatus::SUCCESS) {
    detail = "QnnSampleApp::initializeBackend failed";
    rinCleanupPersistentContext(*session);
    return nullptr;
  }

  stage = "persistent_device_property";
  session->devicePropertySupportStatus = session->app->isDevicePropertySupported();
  if (session->devicePropertySupportStatus != RinPersistentAppStatus::FAILURE) {
    stage = "persistent_create_device";
    session->deviceCreated =
        (session->app->createDevice() == RinPersistentAppStatus::SUCCESS);
    if (!session->deviceCreated) {
      detail = "QnnSampleApp::createDevice failed";
      rinCleanupPersistentContext(*session);
      return nullptr;
    }
  }

  stage = "persistent_profiling";
  if (session->app->initializeProfiling() != RinPersistentAppStatus::SUCCESS) {
    detail = "QnnSampleApp::initializeProfiling failed";
    rinCleanupPersistentContext(*session);
    return nullptr;
  }

  stage = "persistent_op_packages";
  if (session->app->registerOpPackages() != RinPersistentAppStatus::SUCCESS) {
    detail = "QnnSampleApp::registerOpPackages failed";
    rinCleanupPersistentContext(*session);
    return nullptr;
  }

  stage = "persistent_create_from_binary";
  session->contextCreated =
      (session->app->createFromBinary() == RinPersistentAppStatus::SUCCESS);
  if (!session->contextCreated) {
    detail = "QnnSampleApp::createFromBinary failed";
    rinCleanupPersistentContext(*session);
    return nullptr;
  }

  if (session->app->isFinalizeDeserializedGraphSupported() ==
      RinPersistentAppStatus::SUCCESS) {
    stage = "persistent_finalize_graphs";
    if (session->app->finalizeGraphs() != RinPersistentAppStatus::SUCCESS) {
      detail = "QnnSampleApp::finalizeGraphs failed";
      rinCleanupPersistentContext(*session);
      return nullptr;
    }
  }

  return session;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_geniex_demo_image_QnnInProcessNative_runContextPersistent(
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
  const bool nativeInputValue = nativeInput == JNI_TRUE;
  const bool nativeOutputValue = nativeOutput == JNI_TRUE;

  auto elapsedMs = [&]() -> double {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started)
        .count();
  };

  if (backendPath.empty() || systemLibraryPath.empty() || contextPath.empty() ||
      inputListPath.empty() || outputDir.empty()) {
    return makeResult(env,
                      false,
                      "arguments",
                      "missing required path",
                      elapsedMs(),
                      "",
                      nativeLogTail());
  }

  std::call_once(gLogInitOnce, []() {
    gLogReady = qnn::log::initializeLogging(rinQnnLogCallback, QNN_LOG_LEVEL_VERBOSE);
  });
  if (!gLogReady) {
    return makeResult(env,
                      false,
                      "logging",
                      "QNN logging initialization failed",
                      elapsedMs(),
                      "",
                      nativeLogTail());
  }

  const std::string key = rinPersistentKey(backendPath,
                                            systemLibraryPath,
                                            contextPath,
                                            nativeInputValue,
                                            nativeOutputValue);
  auto found = gRinPersistentContexts.find(key);
  const bool cacheHit = found != gRinPersistentContexts.end();
  std::string stage;
  std::string detail;
  RinPersistentContext* session = nullptr;

  if (cacheHit) {
    session = found->second.get();
    stage = "persistent_cache_hit";
  } else {
    if (gRinPersistentContexts.size() >= kRinPersistentContextLimit) {
      return makeResult(env,
                        false,
                        "persistent_cache_capacity",
                        "persistent context cache limit reached",
                        elapsedMs(),
                        "",
                        nativeLogTail());
    }
    auto created = rinCreatePersistentContext(backendPath,
                                               systemLibraryPath,
                                               contextPath,
                                               inputListPath,
                                               outputDir,
                                               nativeInputValue,
                                               nativeOutputValue,
                                               stage,
                                               detail);
    if (!created) {
      return makeResult(env,
                        false,
                        stage,
                        detail,
                        elapsedMs(),
                        "",
                        nativeLogTail());
    }
    session = created.get();
    gRinPersistentContexts.emplace(key, std::move(created));
    stage = "persistent_cache_miss";
  }

  const std::string backendBuild = session->backendBuild;
  auto evictOnFailure = [&](const std::string& failedStage,
                            const std::string& failedDetail) -> jstring {
    const std::string operationLog = nativeLogTail();
    auto it = gRinPersistentContexts.find(key);
    if (it != gRinPersistentContexts.end()) {
      rinCleanupPersistentContext(*it->second);
      gRinPersistentContexts.erase(it);
    }
    return makeResult(env,
                      false,
                      failedStage,
                      failedDetail,
                      elapsedMs(),
                      backendBuild,
                      operationLog);
  };

  stage = "persistent_set_io";
  if (session->app->rinSetRunIO(inputListPath, outputDir) !=
      RinPersistentAppStatus::SUCCESS) {
    return evictOnFailure(stage, session->app->getLastExecutionDetail());
  }

  stage = "persistent_execute";
  if (session->app->executeGraphs(false) != RinPersistentAppStatus::SUCCESS) {
    return evictOnFailure(stage, session->app->getLastExecutionDetail());
  }

  detail = cacheHit ? "persistent_cache_hit" : "persistent_cache_miss_loaded";
  detail += " cached=" + std::to_string(gRinPersistentContexts.size());
  return makeResult(env,
                    true,
                    "complete",
                    detail,
                    elapsedMs(),
                    backendBuild,
                    nativeLogTail());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_geniex_demo_image_QnnInProcessNative_releasePersistentContexts(
    JNIEnv* env,
    jobject /* thiz */) {
  std::lock_guard<std::mutex> lock(gRunMutex);
  const auto started = std::chrono::steady_clock::now();
  const size_t count = gRinPersistentContexts.size();
  rinReleaseAllPersistentContexts();
  const double elapsedMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();
  return makeResult(env,
                    true,
                    "complete",
                    "persistent_released=" + std::to_string(count),
                    elapsedMs,
                    "",
                    "");
}
