//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace wp {
class Stream;

class Graph {
public:
    explicit Graph(Stream& stream);

    ~Graph();

    void capture_begin();

    void end_capture();

    void launch();

private:
    Stream& stream_;
    void* graph_{};
};
}  // namespace wp