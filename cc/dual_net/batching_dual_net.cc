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

  std::vector<DualNet::Result> RunMany(
      std::vector<std::vector<DualNet::BoardFeatures>>&& feature_vecs) {
    std::vector<std::future<DualNet::Result>> futures;
    futures.reserve(feature_vecs.size());

    {
      absl::MutexLock lock(&mutex_);

      size_t old_queue_counter = queue_counter_;
      for (auto& features : feature_vecs) {
        size_t num_features = features.size();
        MG_CHECK(num_features <= static_cast<size_t>(FLAGS_batch_size));

        std::promise<DualNet::Result> promise;
        futures.push_back(std::move(promise.get_future()));

        queue_counter_ += num_features;
        inference_queue_.push({std::move(features), std::move(promise)});
      }
      client_counters_.push(queue_counter_ - old_queue_counter);

      MaybeRunBatches();
    }

    std::vector<DualNet::Result> results;
    results.reserve(futures.size());
    for (auto& future : futures) {
      results.push_back(std::move(future.get()));
    }

    return results;
  }

 private:
  void MaybeRunBatches() EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    while (size_t batch_size =
               std::min(queue_counter_ - run_counter_,
                        static_cast<size_t>(FLAGS_batch_size))) {
      // Stop if we won't fill a batch yet but more request will come.
      if (static_cast<int>(batch_size) < FLAGS_batch_size &&
          num_clients_ > client_counters_.size()) {
        break;
      }

      RunBatch(batch_size);
    }
  }

  void RunBatch(size_t batch_size) EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    std::vector<std::vector<DualNet::BoardFeatures>> features;
    std::vector<std::promise<DualNet::Result>> promises;

    while (batch_size > 0) {
      auto& inference = inference_queue_.front();
      size_t num_features = inference.features.size();

      if (num_features > batch_size) {
        break;  // Request doesn't fit anymore.
      }

      features.push_back(std::move(inference.features));
      promises.push_back(std::move(inference.promise));

      inference_queue_.pop();
      batch_size -= num_features;
      run_counter_ += num_features;

      client_counters_.front() -= num_features;
      if (0 == client_counters_.front()) {
        client_counters_.pop();
      }
    }

    // Unlock the mutex while running inference.
    mutex_.Unlock();
    auto results = dual_net_->RunMany(std::move(features));
    for (size_t i = 0; i < results.size(); ++i) {
      promises[i].set_value(std::move(results[i]));
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
  // Number of features currently in inference queue, per client.
  std::queue<size_t> client_counters_ GUARDED_BY(&mutex_);

  // For printing batching stats in the destructor only.
  size_t num_runs_ GUARDED_BY(&mutex_);
};

class BatchingDualNet : public DualNet {
 public:
  BatchingDualNet(BatchingService* service) : service_(service) {
    service_->IncrementClientCount();
  }

  ~BatchingDualNet() override { service_->DecrementClientCount(); }

  std::vector<Result> RunMany(
      std::vector<std::vector<BoardFeatures>>&& feature_vecs) override {
    return service_->RunMany(std::move(feature_vecs));
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
