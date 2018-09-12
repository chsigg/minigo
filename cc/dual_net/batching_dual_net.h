#ifndef CC_DUAL_NET_BATCHING_DUAL_NET_H_
#define CC_DUAL_NET_BATCHING_DUAL_NET_H_

#include <queue>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "cc/dual_net/dual_net.h"

namespace minigo {

std::unique_ptr<DualNet::Factory> NewBatchingFactory(
    std::unique_ptr<DualNet> dual_net);

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_DUAL_NET_H_
