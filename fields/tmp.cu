#include "geometry.h"
#include "mesh.h"
#include "template_geometry.h"
#include <iostream>

namespace wp::fields {
void test() {
    std::cout << template_geometry_t<Interval, 1>::n_geometry(1) << std::endl;
}

}// namespace wp::fields
