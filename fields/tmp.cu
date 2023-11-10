#include "geometry.h"
#include "mesh.h"
#include "template_geometry.h"
#include <iostream>

namespace wp::fields {
void test() {
    for (auto &pnt : template_geometry_t<Interval, 6>::quadrature_info.pnts) {
        std::cout << pnt[0] << std::endl;
    }
}

}// namespace wp::fields
