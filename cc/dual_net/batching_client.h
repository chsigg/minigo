#ifndef CC_DUAL_NET_BATCHING_CLIENT_H_
#define CC_DUAL_NET_BATCHING_CLIENT_H_

#include <queue>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "cc/dual_net/dual_net.h"

namespace minigo {

std::unique_ptr<DualNet::ClientFactory> NewBatchingClientFactory(
    std::unique_ptr<DualNet> dual_net);

}  // namespace minigo

#endif  // CC_DUAL_NET_BATCHING_CLIENT_H_
