#include "cc/dual_net/factory.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "cc/dual_net/batching_dual_net.h"
#include "gflags/gflags.h"

#ifdef MG_ENABLE_REMOTE_DUAL_NET
#include "cc/dual_net/remote_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "remote"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_REMOTE_DUAL_NET

#ifdef MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "tf"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_TF_DUAL_NET

#ifdef MG_ENABLE_LITE_DUAL_NET
#include "cc/dual_net/lite_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "lite"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_LITE_DUAL_NET

#ifdef MG_ENABLE_TRT_DUAL_NET
#include "cc/dual_net/trt_dual_net.h"
#ifndef MG_DEFAULT_ENGINE
#define MG_DEFAULT_ENGINE "trt"
#endif  // MG_DEFAULT_ENGINE
#endif  // MG_ENABLE_TRT_DUAL_NET

DEFINE_string(engine, MG_DEFAULT_ENGINE,
              "The inference engine to use. Accepted values:"
#ifdef MG_ENABLE_REMOTE_DUAL_NET
              " \"remote\""
#endif
#ifdef MG_ENABLE_TF_DUAL_NET
              " \"tf\""
#endif
#ifdef MG_ENABLE_LITE_DUAL_NET
              " \"lite\""
#endif
#ifdef MG_ENABLE_TRT_DUAL_NET
              " \"trt\""
#endif
);

namespace minigo {
namespace {

std::unique_ptr<DualNet> NewDualNet(const std::string& model_path) {
  if (FLAGS_engine == "remote") {
#ifdef MG_ENABLE_REMOTE_DUAL_NET
    return NewRemoteDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with remote inference support";
#endif  // MG_ENABLE_REMOTE_DUAL_NET
  }

  if (FLAGS_engine == "tf") {
#ifdef MG_ENABLE_TF_DUAL_NET
    return NewTfDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with tf inference support";
#endif  // MG_ENABLE_TF_DUAL_NET
  }

  if (FLAGS_engine == "lite") {
#ifdef MG_ENABLE_LITE_DUAL_NET
    return NewLiteDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with lite inference support";
#endif  // MG_ENABLE_LITE_DUAL_NET
  }

  if (FLAGS_engine == "trt") {
#ifdef MG_ENABLE_TRT_DUAL_NET
    return NewTrtDualNet(model_path);
#else
    MG_FATAL() << "Binary wasn't compiled with TensorRT inference support";
#endif  // MG_ENABLE_TRT_DUAL_NET
  }

  MG_FATAL() << "Unrecognized inference engine \"" << FLAGS_engine << "\"";
  return nullptr;
}

}  // namespace

std::unique_ptr<DualNet::Factory> NewDualNetFactory(
    const std::string& model_path) {
  return NewBatchingFactory(NewDualNet(model_path));
}

}  // namespace minigo
