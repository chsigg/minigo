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

#include "cc/dual_net/tf_dual_net.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "cc/check.h"
#include "cc/constants.h"
#include "cc/thread_safe_queue.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

using tensorflow::DT_FLOAT;
using tensorflow::Env;
using tensorflow::GraphDef;
using tensorflow::NewSession;
using tensorflow::ReadBinaryProto;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

namespace minigo {

namespace {
class TfDualNet : public DualNet {
  class TfWorker {
   public:
    explicit TfWorker(const tensorflow::GraphDef& graph_def) {
      tensorflow::SessionOptions options;
      options.config.mutable_gpu_options()->set_allow_growth(true);
      session_.reset(tensorflow::NewSession(options));
      TF_CHECK_OK(session_->Create(graph_def));

      inputs_.emplace_back(
          "pos_tensor",
          tensorflow::Tensor(tensorflow::DT_FLOAT,
                             tensorflow::TensorShape({FLAGS_batch_size, kN, kN,
                                                      kNumStoneFeatures})));

      output_names_.emplace_back("policy_output");
      output_names_.emplace_back("value_output");
    }

    ~TfWorker() {
      if (session_ != nullptr) {
        TF_CHECK_OK(session_->Close());
      }
    }

    std::vector<Result> RunMany(
        std::vector<std::vector<BoardFeatures>>&& feature_vecs) {
      // Copy the features into the input tensor.
      auto* feature_data = inputs_.front().second.flat<float>().data();
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
      TF_CHECK_OK(session_->Run(inputs_, output_names_, {}, &outputs_));

      // Copy the policies and values from the output tensors.
      const auto* policy_data = outputs_[0].flat<float>().data();
      const auto* value_data = outputs_[1].flat<float>().data();
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
    std::unique_ptr<tensorflow::Session> session_;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;
    std::vector<std::string> output_names_;
    std::vector<tensorflow::Tensor> outputs_;
  };

  struct InferenceData {
    std::vector<std::vector<BoardFeatures>> feature_vecs;
    std::promise<std::vector<Result>> promise;
  };

 public:
  explicit TfDualNet(std::string model_path)
      : DualNet(model_path), running_(true) {
    // If we can't find the specified graph, try adding a .pb extension.
    auto* env = tensorflow::Env::Default();
    if (!env->FileExists(model_path).ok()) {
      model_path = absl::StrCat(model_path, ".pb");
    }

    tensorflow::GraphDef graph_def;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(env, model_path, &graph_def));

    auto functor = [this](const tensorflow::GraphDef& graph_def) {
      TfWorker worker(graph_def);
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

    for (auto device_id : GetGpuIds()) {
      auto device = std::to_string(device_id);
      PlaceOnDevice(&graph_def, "/gpu:" + device);
      // Two threads per device.
      worker_threads_.emplace_back(functor, graph_def);
      worker_threads_.emplace_back(functor, graph_def);
    }
  }

  ~TfDualNet() override {
    running_ = false;
    for (auto& thread : worker_threads_) {
      thread.join();
    }
  }

  std::vector<Result> RunMany(
      std::vector<std::vector<BoardFeatures>>&& feature_vecs) override {
    std::promise<std::vector<Result>> promise;
    auto future = promise.get_future();
    queue_.Push({std::move(feature_vecs), std::move(promise)});
    return future.get();
  }

 private:
  static void PlaceOnDevice(tensorflow::GraphDef* graph_def,
                            const std::string& device) {
    for (auto& node : *graph_def->mutable_node()) {
      if ([&] {
            if (node.op() != "Const") {
              return true;
            }
            auto it = node.attr().find("dtype");
            return it == node.attr().end() ||
                   it->second.type() != tensorflow::DT_INT32;
          }()) {
        node.set_device(device);
      }
    }
  }

  ThreadSafeQueue<InferenceData> queue_;
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> running_;
};
}  // namespace

std::unique_ptr<DualNet> NewTfDualNet(const std::string& model_path) {
  return absl::make_unique<TfDualNet>(model_path);
}

}  // namespace minigo
