LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := rinqnnbridge

SDK_BASE := $(LOCAL_PATH)/../../qnn-native-sdk-2.48/qairt/2.48.0.260626
SAMPLE_SRC := $(SDK_BASE)/examples/QNN/SampleApp/SampleApp/src

LOCAL_C_INCLUDES := \
    $(SDK_BASE)/include/QNN \
    $(SAMPLE_SRC) \
    $(SAMPLE_SRC)/CachingUtil \
    $(SAMPLE_SRC)/Log \
    $(SAMPLE_SRC)/PAL/include \
    $(SAMPLE_SRC)/Utils \
    $(SAMPLE_SRC)/WrapperUtils \
    $(SDK_BASE)/examples/QNN/SampleApp/SampleApp/include/flatbuffers

SAMPLE_CPP := $(wildcard $(SAMPLE_SRC)/*.cpp)
SAMPLE_CPP += $(wildcard $(SAMPLE_SRC)/Log/*.cpp)
SAMPLE_CPP += $(wildcard $(SAMPLE_SRC)/PAL/src/linux/*.cpp)
SAMPLE_CPP += $(wildcard $(SAMPLE_SRC)/PAL/src/common/*.cpp)
SAMPLE_CPP += $(wildcard $(SAMPLE_SRC)/Utils/*.cpp)
SAMPLE_CPP += $(wildcard $(SAMPLE_SRC)/WrapperUtils/*.cpp)
SAMPLE_CPP := $(filter-out $(SAMPLE_SRC)/main.cpp,$(SAMPLE_CPP))

LOCAL_SRC_FILES := rin_qnn_bridge.cpp $(patsubst $(LOCAL_PATH)/%,%,$(SAMPLE_CPP))
LOCAL_CPPFLAGS := -std=c++17 -O3 -Wall -fexceptions -frtti -fvisibility=hidden
LOCAL_LDLIBS := -llog -ldl -lm -lGLESv2 -lEGL

include $(BUILD_SHARED_LIBRARY)
