#include "cc/dual_net/batching_client.h"

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

  // Runs inference on a batch of input features aynchronously.
  DualNet::Result Run(std::vector<DualNet::BoardFeatures>&& features) {
    size_t num_features = features.size();
    MG_CHECK(num_features <= static_cast<size_t>(FLAGS_batch_size));

    std::promise<DualNet::Result> promise;
    auto future = promise.get_future();

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

  void RunBatch(size_t batch_size) {
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
    }

    auto results = dual_net_->RunMany(std::move(features));
    for (size_t i = 0; i < results.size(); ++i) {
      promises[i].set_value(std::move(results[i]));
    }

    ++num_runs_;
  }

  std::unique_ptr<DualNet> dual_net_;

  absl::Mutex mutex_;

  size_t num_clients_ GUARDED_BY(&mutex_);

  std::queue<InferenceData> inference_queue_ GUARDED_BY(&mutex_);
  // Number of features pushed to queue
  size_t queue_counter_ GUARDED_BY(&mutex_);
  // Number of features pushed to dual net.
  size_t run_counter_ GUARDED_BY(&mutex_);

  // For printing batching stats in the destructor only.
  size_t num_runs_ GUARDED_BY(&mutex_);
};

class WeakBatchingClient : public DualNet::Client {
 public:
  WeakBatchingClient(BatchingService* service) : service_(service) {}

  DualNet::Result Run(std::vector<DualNet::BoardFeatures>&& features) override {
    return service_->Run(std::move(features));
  }

 protected:
  BatchingService* service_;
};

class CountedBatchingClient : public WeakBatchingClient {
 public:
  CountedBatchingClient(BatchingService* service)
      : WeakBatchingClient(service) {
    service_->IncrementClientCount();
  }

  ~CountedBatchingClient() override { service_->DecrementClientCount(); }
};

class BatchingClientFactory : public DualNet::ClientFactory {
 public:
  explicit BatchingClientFactory(std::unique_ptr<DualNet> dual_net)
      : service_(std::move(dual_net)) {}

  std::unique_ptr<DualNet::Client> New(bool weak) override {
    if (weak) {
      return absl::make_unique<WeakBatchingClient>(&service_);
    }
    return absl::make_unique<CountedBatchingClient>(&service_);
  }

 private:
  BatchingService service_;
};
}  // namespace

std::unique_ptr<DualNet::ClientFactory> NewBatchingClientFactory(
    std::unique_ptr<DualNet> dual_net) {
  return absl::make_unique<BatchingClientFactory>(std::move(dual_net));
}
}  // namespace minigo
