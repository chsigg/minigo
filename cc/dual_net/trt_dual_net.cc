// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cc/dual_net/trt_dual_net.h"

#include <bitset>
#include <fstream>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/thread_safe_queue.h"
#include "cuda/include/cuda_runtime_api.h"
#include "tensorrt/include/NvInfer.h"
#include "tensorrt/include/NvUffParser.h"

namespace minigo {

namespace {

bool DeviceHasNativeReducedPrecision(int device) {
  cudaDeviceProp props;
  MG_CHECK(cudaGetDeviceProperties(&props, device) == cudaSuccess);
  if (props.major > 6) {
    return true;
  }
  if (props.major == 6) {
    return props.minor != 1;
  }
  if (props.major == 5) {
    return props.minor >= 3;
  }
  return false;
}

class TrtDualNet : public DualNet {
  // TensorRT 4.0.16 ignores the input layout and always assumed NCHW.
  static constexpr auto kInputLayout = nvuffparser::UffInputOrder::kNCHW;

  class TrtLogger : public nvinfer1::ILogger {
   public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          std::cerr << "TensorRT internal error: " << msg << std::endl;
          break;
        case Severity::kERROR:
          std::cerr << "TensorRT error: " << msg << std::endl;
          break;
        case Severity::kWARNING:
          std::cerr << "TensorRT warning: " << msg << std::endl;
          break;
        default:
          break;
      }
    }
  };

  class TrtWorker {
   public:
    explicit TrtWorker(nvinfer1::ICudaEngine* engine) {
      context_ = engine->createExecutionContext();
      MG_CHECK(context_);

      void* host_ptr;
      size_t input_size =
          FLAGS_batch_size * (kN * kN * DualNet::kNumStoneFeatures);
      MG_CHECK(cudaHostAlloc(&host_ptr, input_size * sizeof(float),
                             cudaHostAllocWriteCombined) == cudaSuccess);
      pos_tensor_ = static_cast<float*>(host_ptr);
      size_t output_size = FLAGS_batch_size * (kNumMoves + 1);
      MG_CHECK(cudaHostAlloc(&host_ptr, output_size * sizeof(float),
                             cudaHostAllocDefault) == cudaSuccess);
      value_output_ = static_cast<float*>(host_ptr);
      policy_output_ = value_output_ + FLAGS_batch_size;
    }

    ~TrtWorker() {
      cudaFreeHost(value_output_);
      cudaFreeHost(pos_tensor_);

      context_->destroy();
    }

    Result RunMany(std::vector<BoardFeatures>&& features) {
      size_t num_features = features.size();

      if (kInputLayout == nvuffparser::UffInputOrder::kNCHW) {
        for (auto& feature : features) {
          TransposeBoardFeatures(&feature);
        }
      }

      // Copy the features into the input tensor.
      std::copy_n(features.front().begin(), kNumBoardFeatures * num_features,
                  pos_tensor_);
      // Deallocate features memory.
      std::vector<BoardFeatures>().swap(features);

      // Run the model.
      void* buffers[] = {pos_tensor_, policy_output_, value_output_};
      context_->execute(FLAGS_batch_size, buffers);

      // Copy the policy and value out of the output tensors.
      std::vector<Policy> policies(num_features);
      std::copy_n(policy_output_, kNumMoves * num_features,
                  policies.front().data());

      std::vector<float> values(num_features);
      std::copy_n(value_output_, num_features, values.data());

      return {std::move(policies), std::move(values)};
    }

   private:
    // Transposes features layout in-place from HWC to CHW.
    void TransposeBoardFeatures(BoardFeatures* features) {
      auto* data = features->data();
      std::bitset<kNumBoardFeatures> visited;
      size_t i = 0;
      for (size_t column = 0; column < kNumStoneFeatures; ++column) {
        for (size_t row = 0; row < kN * kN; ++row) {
          float value = data[i];
          while (!visited[i]) {
            visited.set(i);
            // Convert index from row-major to column-major.
            i = i % kNumStoneFeatures * kN * kN + i / kNumStoneFeatures;
            std::swap(value, data[i]);
          }
          ++i;
        }
      }
    }

    nvinfer1::IExecutionContext* context_;

    float* pos_tensor_;
    float* policy_output_;
    float* value_output_;
  };

  struct InferenceData {
    std::vector<BoardFeatures> features;
    std::promise<Result> promise;
  };

 public:
  explicit TrtDualNet(std::string model_path)
      : model_path_(model_path), running_(true) {
    runtime_ = nvinfer1::createInferRuntime(logger_);
    MG_CHECK(runtime_);

    auto* parser = nvuffparser::createUffParser();

    parser->registerInput("pos_tensor",
                          nvinfer1::DimsCHW(DualNet::kNumStoneFeatures, kN, kN),
                          kInputLayout);

    parser->registerOutput("policy_output");
    parser->registerOutput("value_output");

    auto* builder = nvinfer1::createInferBuilder(logger_);
    auto* network = builder->createNetwork();

    if (!std::ifstream(model_path).good()) {
      model_path = absl::StrCat(model_path, ".uff");
    }

    MG_CHECK(
        parser->parse(model_path.c_str(), *network, nvinfer1::DataType::kFLOAT))
        << ". File path: '" << model_path << "'";

    builder->setMaxBatchSize(FLAGS_batch_size);
    builder->setMaxWorkspaceSize(1ull << 30);  // One gigabyte.

    auto gpu_ids = GetGpuIds();
    if (std::all_of(gpu_ids.begin(), gpu_ids.end(),
                    &DeviceHasNativeReducedPrecision)) {
      // All GPUs support fast fp16 math, enable it.
      builder->setFp16Mode(true);
    }

    cudaSetDevice(gpu_ids.front());
    auto* engine = [&] {
      // Building TensorRT engines is not thread-safe.
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      return builder->buildCudaEngine(*network);
    }();
    MG_CHECK(engine);

    network->destroy();
    builder->destroy();
    parser->destroy();

    std::vector<std::future<std::pair<int, nvinfer1::ICudaEngine*>>> futures;
    futures.emplace_back(std::async(std::launch::deferred,
                                    [&] { return std::make_pair(0, engine); }));

    auto* blob = engine->serialize();
    MG_CHECK(blob);

    for (size_t i = 1; i < gpu_ids.size(); ++i) {
      futures.emplace_back(
          std::async(std::launch::async,
                     [&](int device_id) {
                       cudaSetDevice(device_id);
                       return std::make_pair(
                           device_id, runtime_->deserializeCudaEngine(
                                          blob->data(), blob->size(), nullptr));
                     },
                     gpu_ids[i]));
    }

    auto functor = [this](const std::pair<int, nvinfer1::ICudaEngine*>& pair) {
      pthread_setname_np(pthread_self(), "TrtWorker");
      cudaSetDevice(pair.first);
      TrtWorker worker(pair.second);
      while (running_) {
        InferenceData inference;
        if (queue_.PopWithTimeout(&inference, absl::Seconds(1))) {
          auto result = worker.RunMany(std::move(inference.features));
          result.model = model_path_;
          inference.promise.set_value(std::move(result));
        }
      }
    };

    for (auto& future : futures) {
      auto pair = future.get();
      MG_CHECK(pair.second) << "Failed to deserialize TensorRT engine.";
      engines_.push_back(pair.second);
      worker_threads_.emplace_back(functor, pair);
      worker_threads_.emplace_back(functor, pair);
    }
    blob->destroy();
  }

  ~TrtDualNet() override {
    running_ = false;
    for (auto& thread : worker_threads_) {
      thread.join();
    }
    for (auto* engine : engines_) {
      engine->destroy();
    }
    runtime_->destroy();
  }

  Result RunMany(std::vector<BoardFeatures>&& features) override {
    std::promise<Result> promise;
    auto future = promise.get_future();
    queue_.Push({std::move(features), std::move(promise)});
    return future.get();
  }

 private:
  std::string model_path_;

  TrtLogger logger_;
  nvinfer1::IRuntime* runtime_;
  std::vector<nvinfer1::ICudaEngine*> engines_;

  ThreadSafeQueue<InferenceData> queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;
};

}  // namespace

std::unique_ptr<DualNet> NewTrtDualNet(const std::string& model_path) {
  return absl::make_unique<TrtDualNet>(model_path);
}

}  // namespace minigo
