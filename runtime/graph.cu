//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "graph.h"
#include "stream.h"

namespace wp {
Graph::Graph(Stream& stream) : stream_{stream} {}

Graph::~Graph() { check_cuda(cudaGraphExecDestroy((cudaGraphExec_t)graph_)); }

void Graph::capture_begin() {
    check_cuda(cudaStreamBeginCapture((cudaStream_t)stream_.handle(), cudaStreamCaptureModeGlobal));
}

void Graph::end_capture() {
    cudaGraph_t graph = nullptr;
    check_cuda(cudaStreamEndCapture((cudaStream_t)stream_.handle(), &graph));

    if (graph) {
        // enable to create debug GraphVis visualization of graph
        // cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);

        cudaGraphExec_t graph_exec = nullptr;
        // check_cuda(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

        // can use after CUDA 11.4 to permit graphs to capture cudaMallocAsync() operations
        check_cuda(cudaGraphInstantiateWithFlags(&graph_exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));

        // free source graph
        check_cuda(cudaGraphDestroy(graph));

        graph_ = graph_exec;
    } else {
        graph_ = nullptr;
    }
}

void Graph::launch() { check_cuda(cudaGraphLaunch((cudaGraphExec_t)graph_, (cudaStream_t)stream_.handle())); }

}  // namespace wp