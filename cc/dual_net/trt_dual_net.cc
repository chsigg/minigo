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

#include <fstream>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/thread_safe_queue.h"
#include "cuda/cuda_runtime_api.h"
#include "tensorrt/NvInfer.h"
#include "tensorrt/NvUffParser.h"

namespace minigo {

namespace {

class TrtDualNet : public DualNet {
  class TrtLogger : public nvinfer1::ILogger {
   public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override {
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          std::cerr << "TensorRT internal error: " << msg;
          break;
        case Severity::kERROR:
          std::cerr << "TensorRT error: " << msg;
          break;
        case Severity::kWARNING:
          std::cerr << "TensorRT warning: " << msg;
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

    std::vector<Result> RunMany(
        std::vector<std::vector<BoardFeatures>>&& feature_vecs) {
      // Copy the features into the input tensor.
      auto* feature_data = pos_tensor_;
      std::vector<size_t> feature_counts;
      feature_counts.reserve(feature_vecs.size());
      for (const auto& features : feature_vecs) {
        // Copy the features into the input tensor.
        for (const auto& feature : features) {
          feature_data =
              std::copy(feature.begin(), feature.end(), feature_data);
        }
        feature_counts.push_back(features.size());
      }
      // Deallocate feature_vecs memory.
      std::vector<std::vector<BoardFeatures>>().swap(feature_vecs);

      // Run the model.
      void* buffers[] = {pos_tensor_, policy_output_, value_output_};
      context_->execute(FLAGS_batch_size, buffers);

      // Copy the policy and value out of the output tensors.
      const auto* policy_data = policy_output_;
      const auto* value_data = value_output_;
      std::vector<Result> results;
      results.reserve(feature_counts.size());
      for (size_t num_features : feature_counts) {
        std::vector<Policy> policies(num_features);
        std::copy_n(policy_data, kNumMoves * num_features,
                    policies.front().data());
        policy_data += kNumMoves * num_features;

        std::vector<float> values(num_features);
        std::copy_n(value_data, num_features, values.data());
        value_data += num_features;

        results.push_back({std::move(policies), std::move(values)});
      }
      return results;
    }

   private:
    nvinfer1::IExecutionContext* context_;

    float* pos_tensor_;
    float* policy_output_;
    float* value_output_;
  };

  struct InferenceData {
    std::vector<std::vector<BoardFeatures>> feature_vecs;
    std::promise<std::vector<Result>> promise;
  };

 public:
  explicit TrtDualNet(std::string model_path)
      : DualNet(model_path), running_(true) {
    runtime_ = nvinfer1::createInferRuntime(logger_);
    MG_CHECK(runtime_);

    auto* parser = nvuffparser::createUffParser();

    parser->registerInput("pos_tensor",
                          nvinfer1::DimsCHW(DualNet::kNumStoneFeatures, kN, kN),
                          nvuffparser::UffInputOrder::kNCHW);

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
      futures.emplace_back(std::async(std::launch::async, [&](int device_id) {
        cudaSetDevice(device_id);
        return std::make_pair(
            device_id, runtime_->deserializeCudaEngine(blob->data(),
                                                       blob->size(), nullptr));
      }, gpu_ids[i]));
    }

    auto functor = [this](const std::pair<int, nvinfer1::ICudaEngine*>& pair) {
      pthread_setname_np(pthread_self(), "TrtWorker");
      cudaSetDevice(pair.first);
      TrtWorker worker(pair.second);
      while (running_) {
        InferenceData inference;
        if (queue_.PopWithTimeout(&inference, absl::Seconds(1))) {
          auto results = worker.RunMany(std::move(inference.feature_vecs));
          for (auto& result : results) {
            result.model = model_path_;
          }
          inference.promise.set_value(std::move(results));
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

  std::vector<Result> RunMany(
      std::vector<std::vector<BoardFeatures>>&& feature_vecs) override {
    std::promise<std::vector<Result>> promise;
    auto future = promise.get_future();
    queue_.Push({std::move(feature_vecs), std::move(promise)});
    return future.get();
  }

 private:
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
