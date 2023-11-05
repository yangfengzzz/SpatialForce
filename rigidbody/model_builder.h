//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <array>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace wp {
// A helper class for building simulation models at runtime.
//
//    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
//    and builds the scene representation using standard Python data structures (lists),
//    this means it is not differentiable. Once :func:`finalize()`
//    has been called the ModelBuilder transfers all data to Warp tensors and returns
//    an object that may be used for simulation.
class ModelBuilder {
public:
    // particle settings
    static constexpr float default_particle_radius = 0.1;

    // triangle soft mesh settings
    static constexpr float default_tri_ke = 100.0;
    static constexpr float default_tri_ka = 100.0;
    static constexpr float default_tri_kd = 10.0;
    static constexpr float default_tri_drag = 0.0;
    static constexpr float default_tri_lift = 0.0;

    // distance constraint properties
    static constexpr float default_spring_ke = 100.0;
    static constexpr float default_spring_kd = 0.0;

    // edge bending properties
    static constexpr float default_edge_ke = 100.0;
    static constexpr float default_edge_kd = 0.0;

    // rigid shape contact material properties
    static constexpr float default_shape_ke = 1.0e5;
    static constexpr float default_shape_kd = 1000.0;
    static constexpr float default_shape_kf = 1000.0;
    static constexpr float default_shape_mu = 0.5;
    static constexpr float default_shape_restitution = 0.0;
    static constexpr float default_shape_density = 1000.0;

    // joint settings
    static constexpr float default_joint_limit_ke = 100.0;
    static constexpr float default_joint_limit_kd = 1.0;

    // geo settings
    static constexpr float default_geo_thickness = 1e-5;

    int num_envs = 0;
    //
    // particles
    std::vector<float> particle_q;
    std::vector<float> particle_qd;
    std::vector<float> particle_mass;
    std::vector<float> particle_radius;
    std::vector<float> particle_flags;
    float particle_max_velocity = 1e5;

    //
    // shapes (each shape has an entry in these arrays)
    // transform from shape to body
    std::vector<float> shape_transform;
    // maps from shape index to body index
    std::vector<float> shape_body;
    std::vector<float> shape_geo_type;
    std::vector<float> shape_geo_scale;
    std::vector<float> shape_geo_src;
    std::vector<float> shape_geo_is_solid;
    std::vector<float> shape_geo_thickness;
    std::vector<float> shape_material_ke;
    std::vector<float> shape_material_kd;
    std::vector<float> shape_material_kf;
    std::vector<float> shape_material_mu;
    std::vector<float> shape_material_restitution;
    // collision groups within collisions are handled
    std::vector<float> shape_collision_group;
    std::vector<float> shape_collision_group_map;
    int last_collision_group = 0;
    // radius to use for broadphase collision checking
    std::vector<float> shape_collision_radius;
    // whether the shape collides with the ground
    std::vector<float> shape_ground_collision;
    //
    // filtering to ignore certain collision pairs
    std::unordered_set<int> shape_collision_filter_pairs{};
    //
    // geometry
    std::vector<float> geo_meshes;
    std::vector<float> geo_sdfs;
    //
    // springs
    std::vector<float> spring_indices;
    std::vector<float> spring_rest_length;
    std::vector<float> spring_stiffness;
    std::vector<float> spring_damping;
    std::vector<float> spring_control;
    //
    // triangles
    std::vector<float> tri_indices;
    std::vector<float> tri_poses;
    std::vector<float> tri_activations;
    std::vector<float> tri_materials;
    //
    // edges (bending)
    std::vector<float> edge_indices;
    std::vector<float> edge_rest_angle;
    std::vector<float> edge_bending_properties;
    //
    // tetrahedra
    std::vector<float> tet_indices;
    std::vector<float> tet_poses;
    std::vector<float> tet_activations;
    std::vector<float> tet_materials;
    //
    // muscles
    std::vector<float> muscle_start;
    std::vector<float> muscle_params;
    std::vector<float> muscle_activation;
    std::vector<float> muscle_bodies;
    std::vector<float> muscle_points;
    //
    // rigid bodies
    std::vector<float> body_mass;
    std::vector<float> body_inertia;
    std::vector<float> body_inv_mass;
    std::vector<float> body_inv_inertia;
    std::vector<float> body_com;
    std::vector<float> body_q;
    std::vector<float> body_qd;
    std::vector<float> body_name;
    // mapping from body to shapes
    std::vector<float> body_shapes;
    //
    // rigid joints
    std::vector<float> joint;
    // index of the parent body  (constant)
    std::vector<float> joint_parent;
    // mapping from joint to parent bodies
    std::vector<float> joint_parents;
    // index of the child body (constant)
    std::vector<float> joint_child;
    // joint axis in child joint frame (constant)
    std::vector<float> joint_axis;
    // frame of joint in parent (constant)
    std::vector<float> joint_X_p;
    // frame of child com (in child coordinates)  (constant)
    std::vector<float> joint_X_c;
    std::vector<float> joint_q;
    std::vector<float> joint_qd;
    //
    std::vector<float> joint_type;
    std::vector<float> joint_name;
    std::vector<float> joint_armature;
    std::vector<float> joint_target;
    std::vector<float> joint_target_ke;
    std::vector<float> joint_target_kd;
    std::vector<float> joint_axis_mode;
    std::vector<float> joint_limit_lower;
    std::vector<float> joint_limit_upper;
    std::vector<float> joint_limit_ke;
    std::vector<float> joint_limit_kd;
    std::vector<float> joint_act;
    //
    std::vector<float> joint_twist_lower;
    std::vector<float> joint_twist_upper;
    //
    std::vector<float> joint_linear_compliance;
    std::vector<float> joint_angular_compliance;
    std::vector<float> joint_enabled;
    //
    std::vector<float> joint_q_start;
    std::vector<float> joint_qd_start;
    std::vector<float> joint_axis_start;
    std::vector<float> joint_axis_dim;
    std::vector<float> articulation_start;
    //
    int joint_dof_count = 0;
    int joint_coord_count = 0;
    int joint_axis_total_count = 0;
    //
    std::array<float, 3> up_vector;
    std::array<float, 3> up_axis;
    std::array<float, 3> gravity;
    // indicates whether a ground plane has been created
    bool _ground_created{false};
    // constructor parameters for ground plane shape
    struct GroundParams {
        std::array<float, 3> plane{};
        float width{};
        float length{};
        float ke{default_shape_ke};
        float kd{default_shape_kd};
        float kf{default_shape_kf};
        float mu{default_shape_mu};
        float restitution{default_shape_restitution};
    };
    GroundParams _ground_params;

    // Maximum number of soft contacts that can be registered
    int soft_contact_max = 64 * 1024;
    //
    // contacts to be generated within the given distance margin to be generated at
    // every simulation substep (can be 0 if only one PBD solver iteration is used)
    float rigid_contact_margin = 0.1;
    // torsional friction coefficient (only considered by XPBD so far)
    float rigid_contact_torsional_friction = 0.5;
    // rolling friction coefficient (only considered by XPBD so far)
    float rigid_contact_rolling_friction = 0.001;
    //
    // number of rigid contact points to allocate in the model during self.finalize() per environment
    // if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
    int num_rigid_contacts_per_env;

public:
    void shape_count();

    void body_count();

    void joint_count();

    void joint_axis_count();

    void particle_count();

    void tri_count();

    void tet_count();

    void edge_count();

    void spring_count();

    void muscle_count();

    void articulation_count();

    void add_articulation();

    void add_builder();

    void add_body();

    void add_joint();

    void add_joint_revolute();

    void add_joint_prismatic();

    void add_joint_ball();

    void add_joint_fixed();

    void add_joint_free();

    void add_joint_distance();

    void add_joint_universal();

    void add_joint_compound();

    void add_joint_d6();

    void collapse_fixed_joints();

    void add_muscle();

    void add_shape_plane();

    void add_shape_sphere();

    void add_shape_box();

    void add_shape_capsule();

    void add_shape_cylinder();

    void add_shape_cone();

    void add_shape_mesh();

    void add_shape_sdf();

    void _shape_radius();

    void _add_shape();

    void add_particle();

    void add_spring();

    void add_triangle();

    void add_triangles();

    void add_tetrahedron();

    void add_edge();

    void add_edges();

    void add_cloth_grid();

    void add_cloth_mesh();

    void add_particle_grid();

    void add_soft_grid();

    void add_soft_mesh();

    void _update_body_mass();

    void set_ground_plane();

    void _create_ground_plane();

    void finalize();
};
}// namespace wp