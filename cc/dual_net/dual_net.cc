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

#include "cc/dual_net/dual_net.h"

#include <vector>

#include "absl/memory/memory.h"
#include "cc/color.h"
#include "cc/constants.h"
#include "cuda/include/cuda_runtime_api.h"

// TODO(csigg): Expand explanation.
DEFINE_int32(batch_size, 1024, "Inference batch size.");
DEFINE_int32(num_gpus, 0, "Number of GPUs to use.");

namespace minigo {

constexpr int DualNet::kNumStoneFeatures;
constexpr int DualNet::kNumBoardFeatures;

DualNet::Factory::Factory() = default;

DualNet::Factory::~Factory() = default;

DualNet::~DualNet() = default;

void DualNet::SetFeatures(absl::Span<const Position::Stones* const> history,
                          Color to_play, BoardFeatures* features) {
  MG_CHECK(history.size() <= kMoveHistory);
  Color my_color = to_play;
  Color their_color = OtherColor(my_color);

  // Write the features for the position history that we have.
  size_t j = 0;
  for (j = 0; j < history.size(); ++j) {
    auto* dst = features->data() + j * 2;
    const auto* end = dst + kNumBoardFeatures;
    const auto* src = history[j]->data();
    while (dst < end) {
      auto color = src->color();
      ++src;
      dst[0] = color == my_color ? 1 : 0;
      dst[1] = color == their_color ? 1 : 0;
      dst += kNumStoneFeatures;
    }
  }

  // Pad the features with zeros if we have fewer than 8 moves of history.
  for (; j < kMoveHistory; ++j) {
    auto* dst = features->data() + j * 2;
    const auto* end = dst + kNumBoardFeatures;
    while (dst < end) {
      dst[0] = 0;
      dst[1] = 0;
      dst += kNumStoneFeatures;
    }
  }

  // Set the "to play" feature plane.
  float to_play_feature = to_play == Color::kBlack ? 1 : 0;
  auto* dst = features->data() + kPlayerFeature;
  const auto* end = dst + kNumBoardFeatures;
  while (dst < end) {
    dst[0] = to_play_feature;
    dst += kNumStoneFeatures;
  }
}

std::vector<int> GetGpuIds() {
  int num_gpus = FLAGS_num_gpus;
  if (num_gpus == 0) {
    MG_CHECK(cudaGetDeviceCount(&num_gpus) == cudaSuccess);
  }
  std::vector<int> result;
  for (int i = 0; i < num_gpus; ++i) {
    result.push_back(i);
  }
  return result;
}

}  // namespace minigo
