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

#include <unistd.h>
#include <array>
#include <deque>
#include <vector>

#include "cc/position.h"
#include "cc/random.h"
#include "cc/test_utils.h"
#include "gtest/gtest.h"

#if MG_ENABLE_TF_DUAL_NET
#include "cc/dual_net/tf_dual_net.h"
#endif
#if MG_ENABLE_REMOTE_DUAL_NET
#include "cc/dual_net/remote_dual_net.h"
#endif
#if MG_ENABLE_LIGHT_DUAL_NET
#include "cc/dual_net/light_dual_net.h"
#endif
#if MG_ENABLE_TRT_DUAL_NET
#include "cc/dual_net/trt_dual_net.h"
#endif

namespace minigo {
namespace {

using StoneFeatures = DualNet::StoneFeatures;
using BoardFeatures = DualNet::BoardFeatures;

StoneFeatures GetStoneFeatures(const BoardFeatures& features, Coord c) {
  StoneFeatures result;
  for (int i = 0; i < DualNet::kNumStoneFeatures; ++i) {
    result[i] = features[c * DualNet::kNumStoneFeatures + i];
  }
  return result;
}

// Verifies SetFeatures an empty board with black to play.
TEST(DualNetTest, TestEmptyBoardBlackToPlay) {
  Position::Stones stones;
  std::vector<const Position::Stones*> history = {&stones};
  DualNet::BoardFeatures features;
  DualNet::SetFeatures(history, Color::kBlack, &features);

  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (int i = 0; i < DualNet::kPlayerFeature; ++i) {
      EXPECT_EQ(0, f[i]);
    }
    EXPECT_EQ(1, f[DualNet::kPlayerFeature]);
  }
}

// Verifies SetFeatures for an empty board with white to play.
TEST(DualNetTest, TestEmptyBoardWhiteToPlay) {
  Position::Stones stones;
  std::vector<const Position::Stones*> history = {&stones};
  DualNet::BoardFeatures features;
  DualNet::SetFeatures(history, Color::kWhite, &features);

  for (int c = 0; c < kN * kN; ++c) {
    auto f = GetStoneFeatures(features, c);
    for (int i = 0; i < DualNet::kPlayerFeature; ++i) {
      EXPECT_EQ(0, f[i]);
    }
    EXPECT_EQ(0, f[DualNet::kPlayerFeature]);
  }
}

// Verifies SetFeatures.
TEST(DualNetTest, TestSetFeatures) {
  TestablePosition board("");

  std::vector<std::string> moves = {"B9", "H9", "A8", "J9"};
  std::deque<Position::Stones> positions;
  for (const auto& move : moves) {
    board.PlayMove(move);
    positions.push_front(board.stones());
  }

  std::vector<const Position::Stones*> history;
  for (const auto& p : positions) {
    history.push_back(&p);
  }

  DualNet::BoardFeatures features;
  DualNet::SetFeatures(history, board.to_play(), &features);

  //                  B0 W0 B1 W1 B2 W2 B3 W3 B4 W4 B5 W5 B6 W6 B7 W7 C
  StoneFeatures b9 = {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  StoneFeatures h9 = {0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  StoneFeatures a8 = {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  StoneFeatures j9 = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  EXPECT_EQ(b9, GetStoneFeatures(features, Coord::FromString("B9")));
  EXPECT_EQ(h9, GetStoneFeatures(features, Coord::FromString("H9")));
  EXPECT_EQ(a8, GetStoneFeatures(features, Coord::FromString("A8")));
  EXPECT_EQ(j9, GetStoneFeatures(features, Coord::FromString("J9")));
}

// Verfies that features work as expected when capturing.
TEST(DualNetTest, TestStoneFeaturesWithCapture) {
  TestablePosition board("");

  std::vector<std::string> moves = {"J3", "pass", "H2", "J2",
                                    "J1", "pass", "J2"};
  std::deque<Position::Stones> positions;
  for (const auto& move : moves) {
    board.PlayMove(move);
    positions.push_front(board.stones());
  }

  std::vector<const Position::Stones*> history;
  for (const auto& p : positions) {
    history.push_back(&p);
  }

  BoardFeatures features;
  DualNet::SetFeatures(history, board.to_play(), &features);

  //                  W0 B0 W1 B1 W2 B2 W3 B3 W4 B4 W5 B5 W6 B6 W7 B7 C
  StoneFeatures j2 = {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(j2, GetStoneFeatures(features, Coord::FromString("J2")));
}

TEST(DualNetTest, TestBackendsEqual) {
  using Function = std::unique_ptr<DualNet> (*)(const std::string&);
  std::vector<std::pair<std::string, Function>> factories;

#if MG_ENABLE_TF_DUAL_NET
  factories.emplace_back("TfDualNet", &NewTfDualNet);
#endif
#if MG_ENABLE_REMOTE_DUAL_NET
  factories.emplace_back("RemoteDualNet", &NewRemoteDualNet);
#endif
#if MG_ENABLE_LIGHT_DUAL_NET
  factories.emplace_back("LiteDualNet", &NewLiteDualNet);
#endif
#if MG_ENABLE_TRT_DUAL_NET
  factories.emplace_back("TrtDualNet", &NewTrtDualNet);
#endif

  DualNet::BoardFeatures features;
  Random().Uniform(0.0f, 1.0f, absl::MakeSpan(features));

  std::string name;
  DualNet::Policy policy;
  float value = 0.0f;

  auto policy_string = [](const DualNet::Policy& policy) {
    std::ostringstream oss;
    std::copy(policy.begin(), policy.end(),
              std::ostream_iterator<float>(oss, " "));
    return oss.str();
  };

  for (const auto& pair : factories) {
    auto result = pair.second("cc/dual_net/test_model")->RunMany({features});

    if (name.empty()) {
      name = pair.first;
      policy = result.policies.front();
      value = result.values.front();

      continue;
    }

    auto pred = [](float left, float right) {
      return std::abs(left - right) <
             0.0001f * (1.0f + std::abs(left) + std::abs(right));
    };
    EXPECT_EQ(std::equal(policy.begin(), policy.end(),
                         result.policies.front().begin(), pred),
              true)
        << name << ": " << policy_string(policy) << std::endl
        << pair.first << ": " << policy_string(result.policies.front());
    EXPECT_NEAR(value, result.values.front(), 0.0001f)
        << name << " vs " << pair.first;
  }
}  // namespace

}  // namespace
}  // namespace minigo
