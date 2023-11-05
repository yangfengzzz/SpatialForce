//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "device.h"
#include "graph.h"

namespace wp {
Graph::Graph(Device &device, void *graph) : device_{device}, graph_{graph} {}

Graph::~Graph() {
    ContextGuard guard(device_.get_context());

    check_cuda(cudaGraphExecDestroy((cudaGraphExec_t)graph_));
}
}  // namespace wp