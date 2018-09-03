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

#ifndef CC_DUAL_NET_DUAL_NET_H_
#define CC_DUAL_NET_DUAL_NET_H_

#include <future>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "cc/constants.h"
#include "cc/position.h"
#include "gflags/gflags.h"

DECLARE_int32(batch_size);
DECLARE_int32(num_gpus);

namespace minigo {

// The input features to the DualNet neural network have 17 binary feature
// planes. 8 feature planes X_t indicate the presence of the current player's
// stones at time t. A further 8 feature planes Y_t indicate the presence of
// the opposing player's stones at time t. The final feature plane C holds all
// 1s if black is to play, or 0s if white is to play. The planes are
// concatenated together to give input features:
//   [X_t, Y_t, X_t-1, Y_t-1, ..., X_t-7, Y_t-7, C].
class DualNet {
 public:
  // Size of move history in the stone features.
  static constexpr int kMoveHistory = 8;

  // Number of features per stone.
  static constexpr int kNumStoneFeatures = kMoveHistory * 2 + 1;

  // Index of the per-stone feature that describes whether the black or white
  // player is to play next.
  static constexpr int kPlayerFeature = kMoveHistory * 2;

  // Total number of features for the board.
  static constexpr int kNumBoardFeatures = kN * kN * kNumStoneFeatures;

  // TODO(tommadams): Change features type from float to uint8_t.
  using StoneFeatures = std::array<float, kNumStoneFeatures>;
  using BoardFeatures = std::array<float, kNumBoardFeatures>;
  using Policy = std::array<float, kNumMoves>;

  struct Result {
    std::vector<Policy> policies;
    std::vector<float> values;
    std::string model;
  };

  class Service;

  // Base class to perform inferences on a single batch of features.
  class Client {
   public:
    virtual ~Client();

    // Runs inference on a batch of input features. Thread-safe.
    virtual Result RunMany(std::vector<BoardFeatures>&& features) = 0;
  };

  class Service {
   public:
    virtual ~Service();

    // Creates a new client.
    virtual std::unique_ptr<Client> New() = 0;
  };

  explicit DualNet(const std::string& model_path);

  virtual ~DualNet();

  // Runs inference on multiple batches of input features.
  virtual std::vector<Result> RunMany(
      std::vector<std::vector<BoardFeatures>>&& feature_vecs) = 0;

  const std::string& name() const { return model_path_; }

  // Generates the board features from the history of recent moves, where
  // history[0] is the current board position, and history[i] is the board
  // position from i moves ago.
  // history.size() must be <= kMoveHistory.
  // TODO(tommadams): Move Position::Stones out of the Position class so we
  // don't need to depend on position.h.
  static void SetFeatures(absl::Span<const Position::Stones* const> history,
                          Color to_play, BoardFeatures* features);

 protected:
  std::string model_path_;
};

}  // namespace minigo

#endif  // CC_DUAL_NET_DUAL_NET_H_
