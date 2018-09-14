#include "cc/dual_net/batching_dual_net.h"

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "cc/check.h"

namespace minigo {
namespace {
class BatchingService {
  struct InferenceData {
    std::vector<DualNet::BoardFeatures> features;
    std::promise<DualNet::Result> promise;
  };

 public:
  explicit BatchingService(std::unique_ptr<DualNet> dual_net)
      : dual_net_(std::move(dual_net)),
        num_clients_(0),
        queue_counter_(0),
        run_counter_(0),
        num_runs_(0) {}

  ~BatchingService() {
    std::cerr << "Ran " << num_runs_ << " batches with an average size of "
              << static_cast<float>(run_counter_) / num_runs_ << ".\n";
  }

  void IncrementClientCount() {
    absl::MutexLock lock(&mutex_);
    ++num_clients_;
  }

  void DecrementClientCount() {
    absl::MutexLock lock(&mutex_);
    --num_clients_;
    MaybeRunBatches();
  }

  DualNet::Result RunMany(std::vector<DualNet::BoardFeatures>&& features) {
    size_t num_features = features.size();
    MG_CHECK(num_features <= static_cast<size_t>(FLAGS_batch_size));

    std::promise<DualNet::Result> promise;
    std::future<DualNet::Result> future = promise.get_future();

    {
      absl::MutexLock lock(&mutex_);

      queue_counter_ += num_features;
      inference_queue_.push({std::move(features), std::move(promise)});

      MaybeRunBatches();
    }

    return future.get();
  }

 private:
  void MaybeRunBatches() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    while (size_t batch_size =
               std::min(queue_counter_ - run_counter_,
                        static_cast<size_t>(FLAGS_batch_size))) {
      // Stop if we won't fill a batch yet but more request will come.
      if (static_cast<int>(batch_size) < FLAGS_batch_size &&
          num_clients_ > inference_queue_.size()) {
        break;
      }

      RunBatch(batch_size);
    }
  }

  void RunBatch(size_t batch_size) EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    std::vector<DualNet::BoardFeatures> features;
    features.reserve(batch_size);

    std::vector<std::promise<DualNet::Result>> promises;
    std::vector<size_t> feature_counts;

    while (batch_size > 0) {
      auto& inference = inference_queue_.front();
      size_t num_features = inference.features.size();

      if (num_features > batch_size) {
        break;  // Request doesn't fit anymore.
      }

      std::copy_n(inference.features.begin(), num_features,
                  std::back_inserter(features));
      promises.push_back(std::move(inference.promise));

      inference_queue_.pop();
      batch_size -= num_features;
      run_counter_ += num_features;
      feature_counts.push_back(num_features);
    }

    // Unlock the mutex while running inference.
    mutex_.Unlock();
    auto result = dual_net_->RunMany(std::move(features));
    auto policy_it = result.policies.begin();
    auto value_it = result.values.begin();
    auto promise_it = promises.begin();
    for (auto num_features : feature_counts) {
      std::vector<DualNet::Policy> policies(policy_it,
                                            policy_it + num_features);
      policy_it += num_features;
      std::vector<float> values(value_it, value_it + num_features);
      value_it += num_features;
      promise_it->set_value(
          {std::move(policies), std::move(values), result.model});
      ++promise_it;
    }
    mutex_.Lock();

    ++num_runs_;
  }

  std::unique_ptr<DualNet> dual_net_;

  absl::Mutex mutex_;

  size_t num_clients_ GUARDED_BY(&mutex_);

  std::queue<InferenceData> inference_queue_ GUARDED_BY(&mutex_);
  // Number of features pushed to inference queue.
  size_t queue_counter_ GUARDED_BY(&mutex_);
  // Number of features popped from inference queue.
  size_t run_counter_ GUARDED_BY(&mutex_);

  // For printing batching stats in the destructor only.
  size_t num_runs_ GUARDED_BY(&mutex_);
};

class BatchingDualNet : public DualNet {
 public:
  BatchingDualNet(BatchingService* service) : service_(service) {
    service_->IncrementClientCount();
  }

  ~BatchingDualNet() override { service_->DecrementClientCount(); }

  Result RunMany(std::vector<BoardFeatures>&& features) override {
    return service_->RunMany(std::move(features));
  };

 protected:
  BatchingService* service_;
};

class BatchingFactory : public DualNet::Factory {
 public:
  explicit BatchingFactory(std::unique_ptr<DualNet> dual_net)
      : service_(std::move(dual_net)) {}

 private:
  std::unique_ptr<DualNet> New() override {
    return absl::make_unique<BatchingDualNet>(&service_);
  }

  BatchingService service_;
};
}  // namespace

std::unique_ptr<DualNet::Factory> NewBatchingFactory(
    std::unique_ptr<DualNet> dual_net) {
  return absl::make_unique<BatchingFactory>(std::move(dual_net));
}
}  // namespace minigo
