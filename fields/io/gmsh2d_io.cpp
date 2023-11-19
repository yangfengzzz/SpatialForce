//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "gmsh2d_io.h"
#include <fstream>
#include <iostream>
#include <map>
#include <array>
#include <set>

namespace wp::fields {
#define THROW_INVALID_ARG_WITH_MESSAGE_IF(expression, message) \
    if (expression) {                                          \
        throw std::invalid_argument(message);                  \
    }

GmshMesh2D::GmshMesh2D(const std::string &filename) {
    parse_gmsh(filename);
    generate_mesh();
}
GmshMesh2D::~GmshMesh2D() = default;

void GmshMesh2D::parse_gmsh(const std::string &filename) {
    std::ifstream is(filename.c_str());
    std::cerr << "Reading in gmsh data ..." << std::endl;

    // This array stores maps from the 'entities' to the 'physical tags' for
    // points, curves, surfaces and volumes. We use this information later to
    // assign boundary ids.
    std::array<std::map<int, int>, 4> tag_maps;

    std::string line;
    is >> line;
    // first determine file format
    unsigned int gmsh_file_format = 0;
    if (line == "$NOD")
        gmsh_file_format = 10;
    else if (line == "$MeshFormat")
        gmsh_file_format = 20;
    else
        THROW_INVALID_ARG_WITH_MESSAGE_IF(true, "we only treat format > 2.0")

    if (gmsh_file_format == 20) {
        double version;
        unsigned int file_type, data_size;

        is >> version >> file_type >> data_size;

        THROW_INVALID_ARG_WITH_MESSAGE_IF((version < 2.0) || (version > 4.1), "NotImplemented for this gmsh format")
        gmsh_file_format = static_cast<unsigned int>(version * 10);

        THROW_INVALID_ARG_WITH_MESSAGE_IF(file_type != 0, "NotImplemented for this gmsh format")
        THROW_INVALID_ARG_WITH_MESSAGE_IF(data_size != sizeof(double), "NotImplemented for this gmsh format")

        // read the end of the header and the first line of the nodes description
        // to sync ourselves with the format 1 handling above
        is >> line;
        THROW_INVALID_ARG_WITH_MESSAGE_IF(line != "$EndMeshFormat", "Have some problem in Gmsh files")

        is >> line;
        // if the next block is of kind $PhysicalNames, ignore it
        if (line == "$PhysicalNames") {
            do {
                is >> line;
            } while (line != "$EndPhysicalNames");
            is >> line;
        }

        // if the next block is of kind $Entities, parse it
        if (line == "$Entities") {
            unsigned long n_points, n_curves, n_surfaces, n_volumes;

            is >> n_points >> n_curves >> n_surfaces >> n_volumes;
            for (unsigned int i = 0; i < n_points; ++i) {
                // parse point ids
                int tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z;

                // we only care for 'tag' as key for tag_maps[0]
                if (gmsh_file_format > 40) {
                    is >> tag >> box_min_x >> box_min_y >> box_min_z >> n_physicals;
                    box_max_x = box_min_x;
                    box_max_y = box_min_y;
                    box_max_z = box_min_z;
                } else {
                    is >> tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >> box_max_y >> box_max_z >> n_physicals;
                }
                // if there is a physical tag, we will use it as boundary id below
                THROW_INVALID_ARG_WITH_MESSAGE_IF(n_physicals >= 2, "More than one tag is not supported!")
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                    is >> physical_tag;
                tag_maps[0][tag] = physical_tag;
            }
            for (unsigned int i = 0; i < n_curves; ++i) {
                // parse curve ids
                int tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z;

                // we only care for 'tag' as key for tag_maps[1]
                is >> tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >> box_max_y >> box_max_z >> n_physicals;
                // if there is a physical tag, we will use it as boundary id below
                THROW_INVALID_ARG_WITH_MESSAGE_IF(n_physicals >= 2, "More than one tag is not supported!")
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                    is >> physical_tag;
                tag_maps[1][tag] = physical_tag;
                // we don't care about the points associated to a curve, but have
                // to parse them anyway because their format is unstructured
                is >> n_points;
                for (unsigned int j = 0; j < n_points; ++j)
                    is >> tag;
            }

            for (unsigned int i = 0; i < n_surfaces; ++i) {
                // parse surface ids
                int tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z;

                // we only care for 'tag' as key for tag_maps[2]
                is >> tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >> box_max_y >> box_max_z >> n_physicals;
                // if there is a physical tag, we will use it as boundary id below
                THROW_INVALID_ARG_WITH_MESSAGE_IF(n_physicals >= 2, "More than one tag is not supported!")
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                    is >> physical_tag;
                tag_maps[2][tag] = physical_tag;
                // we don't care about the curves associated to a surface, but
                // have to parse them anyway because their format is unstructured
                is >> n_curves;
                for (unsigned int j = 0; j < n_curves; ++j)
                    is >> tag;
            }
            for (unsigned int i = 0; i < n_volumes; ++i) {
                // parse volume ids
                int tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z;

                // we only care for 'tag' as key for tag_maps[3]
                is >> tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >> box_max_y >> box_max_z >> n_physicals;
                // if there is a physical tag, we will use it as boundary id below
                THROW_INVALID_ARG_WITH_MESSAGE_IF(n_physicals >= 2, "More than one tag is not supported!")
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                    is >> physical_tag;
                tag_maps[3][tag] = physical_tag;
                // we don't care about the surfaces associated to a volume, but
                // have to parse them anyway because their format is unstructured
                is >> n_surfaces;
                for (unsigned int j = 0; j < n_surfaces; ++j)
                    is >> tag;
            }
            is >> line;
            THROW_INVALID_ARG_WITH_MESSAGE_IF(line != "$EndEntities", "Have some problem in Gmsh files")
            is >> line;
        }

        // if the next block is of kind $PartitionedEntities, ignore it
        if (line == "$PartitionedEntities") {
            do {
                is >> line;
            } while (line != "$EndPartitionedEntities");
            is >> line;
        }

        // but the next thing should,
        // in any case, be the list of
        // nodes:
        THROW_INVALID_ARG_WITH_MESSAGE_IF(line != "$Nodes", "Have some problem in Gmsh files")
    }

    // now read the nodes list
    unsigned int n_vertices;
    unsigned int n_cells;
    unsigned int dummy;
    int n_entity_blocks = 1;
    if (gmsh_file_format > 40) {
        int min_node_tag;
        int max_node_tag;
        is >> n_entity_blocks >> n_vertices >> min_node_tag >> max_node_tag;
    } else if (gmsh_file_format == 40) {
        is >> n_entity_blocks >> n_vertices;
    } else
        is >> n_vertices;

    std::cerr << "\tReading nodes data ..." << std::endl;
    pnt.resize(n_vertices);
    // set up mapping between numbering
    // in msh-file (nod) and in the
    // vertices vector
    std::map<int, int> vertex_indices;

    {
        unsigned int global_vertex = 0;
        for (int entity_block = 0; entity_block < n_entity_blocks; ++entity_block) {
            int parametric;
            unsigned long numNodes;

            if (gmsh_file_format < 40) {
                numNodes = n_vertices;
                parametric = 0;
            } else {
                // for gmsh_file_format 4.1 the order of tag and dim is reversed,
                // but we are ignoring both anyway.
                int tagEntity, dimEntity;
                is >> tagEntity >> dimEntity >> parametric >> numNodes;
            }

            std::vector<int> vertex_numbers;
            int vertex_number;
            if (gmsh_file_format > 40)
                for (unsigned long vertex_per_entity = 0; vertex_per_entity < numNodes; ++vertex_per_entity) {
                    is >> vertex_number;
                    vertex_numbers.push_back(vertex_number);
                }

            for (unsigned long vertex_per_entity = 0; vertex_per_entity < numNodes; ++vertex_per_entity, ++global_vertex) {
                int vertex_number;
                double x[3];

                // read vertex
                if (gmsh_file_format > 40) {
                    vertex_number = vertex_numbers[vertex_per_entity];
                    is >> x[0] >> x[1] >> x[2];
                } else
                    is >> vertex_number >> x[0] >> x[1] >> x[2];

                for (unsigned int d = 0; d < dim; ++d)
                    pnt[global_vertex][d] = x[d];
                // store mapping
                vertex_indices[vertex_number] = global_vertex;

                // ignore parametric coordinates
                if (parametric != 0) {
                    double u = 0.;
                    double v = 0.;
                    is >> u >> v;
                    (void)u;
                    (void)v;
                }
            }
        }
        THROW_INVALID_ARG_WITH_MESSAGE_IF(global_vertex != n_vertices, "DimensionMismatch")
    }

    // Assert we reached the end of the block
    is >> line;
    static const std::string end_nodes_marker[] = {"$ENDNOD", "$EndNodes"};
    THROW_INVALID_ARG_WITH_MESSAGE_IF(line != end_nodes_marker[gmsh_file_format == 10 ? 0 : 1],
                                      "Have some problem in Gmsh files")

    // Now read in next bit
    is >> line;
    static const std::string begin_elements_marker[] = {"$ELM", "$Elements"};
    THROW_INVALID_ARG_WITH_MESSAGE_IF(line != begin_elements_marker[gmsh_file_format == 10 ? 0 : 1],
                                      "Have some problem in Gmsh files")

    // now read the cell list
    if (gmsh_file_format > 40) {
        int min_node_tag;
        int max_node_tag;
        is >> n_entity_blocks >> n_cells >> min_node_tag >> max_node_tag;
    } else if (gmsh_file_format == 40) {
        is >> n_entity_blocks >> n_cells;
    } else {
        n_entity_blocks = 1;
        is >> n_cells;
    }

    std::cerr << "\tReading geometry data ..." << std::endl;
    ele.resize(n_cells);
    {
        unsigned int global_cell = 0;
        unsigned int global_surf = 0;
        for (int entity_block = 0; entity_block < n_entity_blocks; ++entity_block) {
            unsigned int material_id;
            unsigned long numElements;
            int cell_type;

            if (gmsh_file_format < 40) {
                material_id = 0;
                cell_type = 0;
                numElements = n_cells;
            } else if (gmsh_file_format == 40) {
                int tagEntity, dimEntity;
                is >> tagEntity >> dimEntity >> cell_type >> numElements;
                material_id = tag_maps[dimEntity][tagEntity];
            } else {
                // for gmsh_file_format 4.1 the order of tag and dim is reversed,
                int tagEntity, dimEntity;
                is >> dimEntity >> tagEntity >> cell_type >> numElements;
                material_id = tag_maps[dimEntity][tagEntity];
            }

            for (unsigned int cell_per_entity = 0; cell_per_entity < numElements; ++cell_per_entity, ++global_cell) {
                // note that since in the input
                // file we found the number of
                // cells at the top, there
                // should still be input here,
                // so check this:
                // AssertThrow(in, ExcIO());

                unsigned int nod_num = 0;

                // For file format version 1, the format of each cell is as follows:
                // elm-number elm-type reg-phys reg-elem number-of-nodes
                // node-number-list
                //
                // However, for version 2, the format reads like this:
                // elm-number elm-type number-of-tags < tag > ... node-number-list
                //
                // For version 4, we have:
                // tag(int) numVert(int) ...
                //
                // In the following, we will ignore the element number (we simply
                // enumerate them in the order in which we read them, and we will
                // take reg-phys (version 1) or the first tag (version 2, if any tag
                // is given at all) as material id. For version 4, we already read
                // the material and the cell type in above.

                unsigned int elm_number = 0;
                if (gmsh_file_format < 40) {
                    is >> elm_number // ELM-NUMBER
                        >> cell_type;// ELM-TYPE
                }

                if (gmsh_file_format < 20) {
                    is >> material_id// REG-PHYS
                        >> dummy     // reg_elm
                        >> nod_num;
                } else if (gmsh_file_format < 40) {
                    // read the tags; ignore all but the first one which we will
                    // interpret as the material_id (for cells) or boundary_id
                    // (for faces)
                    unsigned int n_tags;
                    is >> n_tags;
                    if (n_tags > 0)
                        is >> material_id;
                    else
                        material_id = 0;

                    for (unsigned int i = 1; i < n_tags; ++i)
                        is >> dummy;

                } else {
                    // ignore tag
                    int tag;
                    is >> tag;
                }

                //`ELM-TYPE'
                // defines the geometrical type of the N-th element:
                // `1'
                // Line (2 nodes, 1 edge).
                //
                // `3'
                // Quadrangle (4 nodes, 4 edges).
                //
                // `5'
                // Hexahedron (8 nodes, 12 edges, 6 faces).
                //
                // `15'
                // Point (1 node).
                GeometryBM geo;
                int node_index = 0;
                switch (cell_type) {
                    case POINT:
                        // read the indices of nodes given
                        if (gmsh_file_format < 20) {
                            for (unsigned int i = 0; i < nod_num; ++i)
                                is >> node_index;
                        } else {
                            is >> node_index;
                        }

                        geo.vtx[0] = vertex_indices[node_index];
                        nodes.push_back(geo);
                        break;
                    case LINE:
                        geo.bm = int(material_id);// geometry region(material marker)
                        geo.vtx.resize(N_LINE_NODE);
                        for (int l = 0; l < N_LINE_NODE; l++) {// index of node
                            is >> node_index;
                            geo.vtx[l] = vertex_indices[node_index];
                        }
                        lines.push_back(geo);
                        break;
                    case TRIANGLE:
                        ele[global_surf].vertex.resize(N_TRIANGLE_NODE);
                        for (int l = 0; l < N_TRIANGLE_NODE; l++) {// index of node
                            is >> node_index;
                            ele[global_surf].vertex[l] = vertex_indices[node_index];
                        }
                        global_surf++;
                        break;
                    default:
                        THROW_INVALID_ARG_WITH_MESSAGE_IF(true, "Unsupported such type of element")
                        break;
                }
            }
        }
        ele.resize(global_surf);
    }

    // Assert we reached the end of the block
    is >> line;
    static const std::string end_elements_marker[] = {"$ENDELM", "$EndElements"};
    THROW_INVALID_ARG_WITH_MESSAGE_IF(line != end_elements_marker[gmsh_file_format == 10 ? 0 : 1],
                                      "Have some problem in Gmsh files")
}

void GmshMesh2D::base_generate_mesh() {
    int i, j, k, l;
    int32_t p;
    std::cerr << "Generate mesh structure from the simplest mesh ..." << std::endl;

    size_t n = n_element();
    std::vector<std::vector<size_t>> pnt_patch(n_point());
    for (i = 0; i < n; ++i) {
        for (j = 0; j < element(i).vertex.size(); ++j) {
            pnt_patch[element_vertex(i, j)].push_back(i);
        }
    }
    std::vector<std::set<size_t, std::less<>>> ele_patch(n);
    for (i = 0; i < n; ++i) {
        for (j = 0; j < element(i).vertex.size(); ++j) {
            ele_patch[i].insert(pnt_patch[element_vertex(i, j)].begin(), pnt_patch[element_vertex(i, j)].end());
        }
    }
    pnt_patch.clear();

    std::vector<std::vector<std::vector<int32_t>>> ele_geo(n, std::vector<std::vector<int32_t>>(dim + 1));

    GeometryBM g;
    point() = pnt;
    for (i = 0; i <= dim; i++)
        geometry((int)i).clear();
    for (i = 0, p = -1; i < n; i++) {
        std::vector<std::vector<int32_t>> &geo_img = ele_geo[i];

        geo_img[0].resize(base_template_geometry_t::n_points(), -1);
        g.vertex().resize(1);
        g.boundary().resize(1);
        for (j = 0; j < base_template_geometry_t::n_points(); j++) {
            g.vertex(0) = element_vertex(i, j);
            g.boundary(0) = element_vertex(i, j);

            bool is_found = false;
            int geo_idx = 0;
            auto the_ele = ele_patch[i].begin(), end_ele = ele_patch[i].end();
            for (; the_ele != end_ele; ++the_ele) {
                size_t ele_idx = *the_ele;
                if (ele_idx >= i)
                    continue;
                for (l = 0; l < ele_geo[ele_idx][0].size(); ++l) {
                    geo_idx = ele_geo[ele_idx][0][l];
                    if (geo_idx >= 0 && geometry(0, geo_idx).vertex(0) == g.vertex(0)) {
                        is_found = true;
                        break;
                    }
                }
                if (is_found)
                    break;
            }
            if (!is_found) {
                geo_idx = n_geometry(0);
                g.index() = geo_idx;
                geometry(0).push_back(g);
            }
            geo_img[0][j] = geo_idx;
        }
        for (j = 1; j <= dim; j++) {
            geo_img[j].resize(base_template_geometry_t::n_geometry((int)j));
            for (k = 0; k < base_template_geometry_t::n_geometry((int)j); k++) {
                g.vertex().resize(base_template_geometry_t::n_geometry_vertex((int)j, k));
                g.boundary().resize(base_template_geometry_t::n_geometry_boundary((int)j, k));
                for (l = 0; l < g.n_vertex(); l++)
                    g.vertex(l) = geo_img[0][base_template_geometry_t::geometry_vertex((int)j, k, l)];
                for (l = 0; l < g.n_boundary(); l++)
                    g.boundary(l) = geo_img[j - 1][base_template_geometry_t::geometry_boundary((int)j, k, l)];
                bool is_found = false;
                int geo_idx = 0;
                auto the_ele = ele_patch[i].begin(), end_ele = ele_patch[i].end();
                for (; the_ele != end_ele; ++the_ele) {
                    size_t ele_idx = *the_ele;
                    if (ele_idx >= i)
                        continue;
                    for (l = 0; l < ele_geo[ele_idx][j].size(); ++l) {
                        geo_idx = ele_geo[ele_idx][j][l];
                        if (geo_idx >= 0 && is_same(geometry((int)j, geo_idx), g)) {
                            is_found = true;
                            break;
                        }
                    }
                    if (is_found)
                        break;
                }

                if (!is_found) {
                    geo_idx = n_geometry((int)j);
                    g.index() = geo_idx;
                    geometry((int)j).push_back(g);
                }
                geo_img[j][k] = geo_idx;
            }
        }
        if (static_cast<int32_t>(100 * i / n) > p) {
            p = 100 * i / n;
            std::cerr << "\r" << p << "% OK!" << std::flush;
        }
    }
    std::cerr << "\r";

    for (j = 1; j <= dim; j++) {
        for (k = 0; k < n_geometry((int)j); k++) {
            GeometryBM &geo = geometry((int)j, k);
            for (l = 0; l < geo.n_vertex(); l++) {
                n = geo.vertex(l);
                geo.vertex(l) = geometry(0, n).vertex(0);
            }
        }
    }
    for (k = 0; k < n_geometry(1); k++) {
        GeometryBM &geo = geometry(1, k);
        for (l = 0; l < geo.n_boundary(); l++) {
            n = geo.boundary(l);
            geo.boundary(l) = geometry(0, n).vertex(0);
        }
    }
    for (k = 0; k < n_geometry(0); k++) {
        geometry(0, k).vertex(0) = k;
        geometry(0, k).boundary(0) = k;
    }
}

void GmshMesh2D::generate_mesh() {
    base_generate_mesh();

    int i, j;
    std::vector<size_t> index(n_geometry(0));
    for (i = 0; i < n_geometry(0); i++)
        index[geometry(0, i).vertex(0)] = i;

    std::list<GeometryBM>::iterator the_geo, end_geo;
    the_geo = lines.begin();
    end_geo = lines.end();
    for (; the_geo != end_geo; the_geo++) {
        for (i = 0; i < the_geo->n_vertex(); i++) {
            j = the_geo->vertex(i);
            the_geo->vertex(i) = index[j];
        }
    }
    the_geo = nodes.begin();
    end_geo = nodes.end();
    for (; the_geo != end_geo; the_geo++) {
        for (i = 0; i < the_geo->n_vertex(); i++) {
            j = the_geo->vertex(i);
            the_geo->vertex(i) = index[j];
        }
    }

    the_geo = lines.begin();
    end_geo = lines.end();
    for (; the_geo != end_geo; the_geo++) {
        for (i = 0; i < n_geometry(1); i++) {
            if (is_same(geometry(1, i), *the_geo)) {
                boundaryMark(1, i) = the_geo->boundaryMark();
                break;
            }
        }
    }
    for (i = 0; i < n_geometry(1); i++) {
        if (boundaryMark(1, i) == 0)
            continue;
        for (j = 0; j < geometry(1, i).n_boundary(); j++) {
            boundaryMark(0, geometry(1, i).boundary(j)) = boundaryMark(1, i);
        }
    }
    the_geo = nodes.begin();
    end_geo = nodes.end();
    for (; the_geo != end_geo; the_geo++) {
        for (i = 0; i < n_geometry(0); i++) {
            if (is_same(geometry(0, i), *the_geo)) {
                boundaryMark(0, i) = the_geo->boundaryMark();
                break;
            }
        }
    }
}

}// namespace wp::fields