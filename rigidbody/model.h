//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/array.h"
#include "core/vec.h"
#include "core/mat.h"
#include <vector>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <array>

namespace wp {
constexpr uint32_t PARTICLE_FLAG_ACTIVE = 1 << 0;

// Shape geometry types
enum class GeometryType {
    GEO_SPHERE,
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CYLINDER,
    GEO_CONE,
    GEO_MESH,
    GEO_SDF,
    GEO_PLANE,
    GEO_NONE
};

// Types of joints linking rigid bodies
enum class JointType {
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_BALL,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_COMPOUND,
    JOINT_UNIVERSAL,
    JOINT_DISTANCE,
    JOINT_D6,
};

// Joint axis mode types
enum class JointMode {
    JOINT_MODE_LIMIT,
    JOINT_MODE_TARGET_POSITION,
    JOINT_MODE_TARGET_VELOCITY
};

// Material properties pertaining to rigid shape contact dynamics
struct ModelShapeMaterials {
    // The contact elastic stiffness (only used by Euler integrator)
    array_t<float> ke;
    // The contact damping stiffness (only used by Euler integrator)
    array_t<float> kd;
    // The contact friction stiffness (only used by Euler integrator)
    array_t<float> kf;
    // The coefficient of friction
    array_t<float> mu;
    // The coefficient of restitution (only used by XPBD integrator)
    array_t<float> restitution;
};

struct ModelShapeGeometry {
    // The type of geometry (GEO_SPHERE, GEO_BOX, etc.)
    array_t<int32_t> type;
    // Indicates whether the shape is solid or hollow
    array_t<uint8_t> is_solid;
    // The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)
    array_t<float> thickness;
    // Pointer to the source geometry (in case of a mesh, zero otherwise)
    array_t<uint64_t> source;
    // The 3D scale of the shape
    array_t<vec3f> scale;
};

// Axis (linear or angular) of a joint that can have bounds and be driven towards a target
struct JointAxis {
    float limit_lower;
    float limit_upper;
    float limit_ke;
    float limit_kd;
    float target;
    float target_ke;
    float target_kd;
    JointMode mode;
};

// Describes a signed distance field for simulation
struct SDF {
    void *volume{};

    bool has_inertia{true};
    bool is_solid{true};
    float mass{};
    float com{};
    float I{};

    void finalize();
};

// Describes a triangle collision mesh for simulation
//struct Mesh {
//    std::vector<vec3f> vertices;
//    std::vector<int32_t> indices;
//    bool is_solid;
//    bool has_inertia;
//
//    float mass{1.0};
//    mat33f I{};
//
//    void finalize();
//};

/// The State object holds all *time-varying* data for a model.
//
//    Time-varying data includes particle positions, velocities, rigid body states, and
//    anything that is output from the integrator as derived data, e.g.: forces.
//
//    The exact attributes depend on the contents of the model. State objects should
//    generally be created using the :func:`Model.state()` function.
struct State {
    int particle_count{};
    int body_count{};
    array_t<vec3f> particle_q;
    array_t<vec3f> particle_qd;
    array_t<vec3f> particle_f;

    array_t<vec3f> body_q;
    array_t<vec3f> body_qd;
    array_t<vec3f> body_f;

    void clear_forces();
    void flatten();
};

std::tuple<float, vec3f, mat33f> compute_shape_mass(GeometryType type, float scale, float density, bool is_solid, float thickness) {
    if (density == 0 || type == GeometryType::GEO_PLANE) {
        return std::make_tuple(0, vec3f(), mat33f());
    }

    if (type == GeometryType::GEO_SPHERE) {

    } else if (type == GeometryType::GEO_BOX) {

    } else if (type == GeometryType::GEO_CAPSULE) {
    } else if (type == GeometryType::GEO_CYLINDER) {

    } else if (type == GeometryType::GEO_CONE) {
    }

    return std::make_tuple(0, vec3f(), mat33f());
}

std::tuple<float, vec3f, mat33f> compute_shape_mass(GeometryType type, SDF &src, float scale, float density, bool is_solid, float thickness) {
    if (src.has_inertia && src.mass > 0 && src.is_solid == is_solid) {
        auto m = src.mass;
        auto c = src.com;
        auto I = src.I;
    }

    return std::make_tuple(0, vec3f(), mat33f());
}

//std::tuple<float, vec3f, mat33f> compute_shape_mass(GeometryType type, Mesh &src, float scale, float density, bool is_solid, float thickness) {
//    return std::make_tuple(0, vec3f(), mat33f());
//}

// Holds the definition of the simulation model
//
//    This class holds the non-time varying description of the system, i.e.:
//    all geometry, constraints, and parameters used to describe the simulation.
class Model {
    //  Number of articulation environments that were added to the ModelBuilder via `add_builder`
    int num_envs{0};
    // Particle positions, shape [particle_count, 3], float
    array_t<float> particle_q;
    // Particle velocities, shape [particle_count, 3], float
    array_t<float> particle_qd;
    // Particle mass
    array_t<float> particle_mass;
    // Particle inverse mass
    array_t<float> particle_inv_mass;
    // Particle radius
    array_t<float> _particle_radius;
    // Maximum particle radius (useful for HashGrid construction)
    float particle_max_radius = 0.0;
    // Particle normal contact stiffness (used by SemiImplicitIntegrator)
    float particle_ke = 1.0e3;
    // Particle normal contact damping (used by SemiImplicitIntegrator)
    float particle_kd = 1.0e2;
    // Particle friction force stiffness (used by SemiImplicitIntegrator)
    float particle_kf = 1.0e2;
    // Particle friction coefficient
    float particle_mu = 0.5;
    //  Particle cohesion strength
    float particle_cohesion = 0.0;
    // Particle adhesion strength
    float particle_adhesion = 0.0;
    // HashGrid instance used for accelerated simulation of particle interactions
    //    HashGrid particle_grid;
    // Particle enabled state
    array_t<bool> particle_flags;
    // Maximum particle velocity (to prevent instability)
    float particle_max_velocity = 1e5;

    // Rigid shape transforms
    array_t<float> shape_transform;
    // Rigid shape body index
    array_t<int> shape_body;
    // Mapping from body index to list of attached shape indices
    std::unordered_map<int, int> body_shapes;
    // Rigid shape contact materials
    ModelShapeMaterials shape_materials;
    // Shape geometry properties (geo type, scale, thickness, etc.)
    ModelShapeGeometry shape_geo;
    // List of `wp.Mesh` instances used for rendering of mesh geometry
//    Mesh shape_geo_src;

    // Collision group of each shape
    std::vector<int> shape_collision_group;
    // Mapping from collision group to list of shape indices
    std::unordered_map<int, int> shape_collision_group_map;
    // Pairs of shape indices that should not collide
    std::unordered_set<int> shape_collision_filter_pairs;
    // Collision radius of each shape used for bounding sphere broadphase collision checking
    array_t<float> shape_collision_radius;
    // Indicates whether each shape should collide with the ground
    std::vector<int> shape_ground_collision;
    // Pairs of shape indices that may collide
    array_t<int> shape_contact_pairs;
    // Pairs of shape, ground indices that may collide
    array_t<int> shape_ground_contact_pairs;

    // Particle spring indices
    array_t<int> spring_indices;
    // Particle spring rest length
    array_t<float> spring_rest_length;
    // Particle spring stiffness
    array_t<float> spring_stiffness;
    // Particle spring damping
    array_t<float> spring_damping;
    // Particle spring activation
    array_t<float> spring_control;

    // Triangle element indices
    array_t<int> tri_indices;
    // Triangle element rest pose
    array_t<float> tri_poses;
    // Triangle element activations
    array_t<float> tri_activations;
    // Triangle element materials
    array_t<float> tri_materials;

    // Bending edge indices
    array_t<int> edge_indices;
    // Bending edge rest angle
    array_t<float> edge_rest_angle;
    // Bending edge stiffness and damping parameters
    array_t<float> edge_bending_properties;

    // Tetrahedral element indices
    array_t<int> tet_indices;
    // Tetrahedral rest poses
    array_t<float> tet_poses;
    // Tetrahedral volumetric activations
    array_t<float> tet_activations;
    // Tetrahedral elastic parameters in form :math:`k_{mu}, k_{lambda}, k_{damp}`
    array_t<float> tet_materials;

    // Poses of rigid bodies used for state initialization
    array_t<float> body_q;
    // Velocities of rigid bodies used for state initialization
    array_t<float> body_qd;
    // Rigid body center of mass (in local frame)
    array_t<float> body_com;
    // Rigid body inertia tensor (relative to COM)
    array_t<float> body_inertia;
    // Rigid body inverse inertia tensor (relative to COM)
    array_t<float> body_inv_inertia;
    // Rigid body mass
    array_t<float> body_mass;
    // Rigid body inverse mass
    array_t<float> body_inv_mass;
    // Rigid body names
    std::vector<std::string> body_name;

    // Generalized joint positions used for state initialization
    array_t<float> joint_q;
    // Generalized joint velocities used for state initialization
    array_t<float> joint_qd;
    // Generalized joint actuation force
    array_t<float> joint_act;
    // Joint type
    array_t<int> joint_type;
    // Joint parent body indices
    array_t<int> joint_parent;
    // Joint child body indices
    array_t<int> joint_child;
    // Joint transform in parent frame
    array_t<float> joint_X_p;
    // Joint mass frame in child frame
    array_t<float> joint_X_c;
    // Joint axis in child frame
    array_t<float> joint_axis;
    // Armature for each joint
    array_t<float> joint_armature;
    // Joint target position/velocity (depending on joint axis mode)
    array_t<float> joint_target;
    // Joint stiffness
    array_t<float> joint_target_ke;
    // Joint damping
    array_t<float> joint_target_kd;
    // Start index of the first axis per joint
    array_t<int> joint_axis_start;
    // Number of linear and angular axes per joint
    array_t<int> joint_axis_dim;
    // Joint axis mode
    array_t<int> joint_axis_mode;
    // Joint linear compliance
    array_t<float> joint_linear_compliance;
    // Joint angular compliance
    array_t<float> joint_angular_compliance;
    // Joint enabled
    array_t<int> joint_enabled;
    // Joint lower position limits
    array_t<float> joint_limit_lower;
    // Joint upper position limits
    array_t<float> joint_limit_upper;
    // Joint position limit stiffness (used by SemiImplicitIntegrator)
    array_t<float> joint_limit_ke;
    // Joint position limit damping (used by SemiImplicitIntegrator)
    array_t<float> joint_limit_kd;
    // Joint lower twist limit
    array_t<float> joint_twist_lower;
    // Joint upper twist limit
    array_t<float> joint_twist_upper;
    // Start index of the first position coordinate per joint
    array_t<int> joint_q_start;
    // Start index of the first velocity coordinate per joint
    array_t<int> joint_qd_start;
    // Articulation start index
    array_t<int> articulation_start;
    // Joint names
    std::vector<std::string> joint_name;
    // Joint attachment force stiffness
    float joint_attach_ke = 1.0e3;
    // Joint attachment force damping
    float joint_attach_kd = 1.0e2;

    // Contact margin for generation of soft contacts
    float soft_contact_margin = 0.2;
    // Stiffness of soft contacts (used by SemiImplicitIntegrator)
    float soft_contact_ke = 1.0e3;
    // Damping of soft contacts (used by SemiImplicitIntegrator)
    float soft_contact_kd = 10.0;
    // Stiffness of friction force in soft contacts (used by SemiImplicitIntegrator)
    float soft_contact_kf = 1.0e3;
    // Friction coefficient of soft contacts
    float soft_contact_mu = 0.5;
    // Restitution coefficient of soft contacts (used by XPBDIntegrator)
    float soft_contact_restitution = 0.0;

    // Contact margin for generation of rigid body contacts
    float rigid_contact_margin;
    // Torsional friction coefficient for rigid body contacts (used by XPBDIntegrator)
    float rigid_contact_torsional_friction;
    // Rolling friction coefficient for rigid body contacts (used by XPBDIntegrator)
    float rigid_contact_rolling_friction;

    // Whether the ground plane and ground contacts are enabled
    bool ground{true};
    // Ground plane 3D normal and offset
    array_t<float> ground_plane;
    // Up vector of the world
    std::array<float, 3> up_vector{0.0, 1.0, 0.0};
    // Up axis
    int up_axis = 1;
    // Gravity vector
    std::array<float, 3> gravity{0.0, -9.81, 0.0};

    // Total number of particles in the system
    int particle_count = 0;
    // Total number of bodies in the system
    int body_count = 0;
    // Total number of shapes in the system
    int shape_count = 0;
    // Total number of joints in the system
    int joint_count = 0;
    // Total number of joint axes in the system
    int joint_axis_count = 0;
    // Total number of triangles in the system
    int tri_count = 0;
    // Total number of tetrahedra in the system
    int tet_count = 0;
    // Total number of edges in the system
    int edge_count = 0;
    // Total number of springs in the system
    int spring_count = 0;
    // Total number of muscles in the system
    int muscle_count = 0;
    // Total number of articulations in the system
    int articulation_count = 0;
    // Total number of velocity degrees of freedom of all joints in the system
    int joint_dof_count = 0;
    // Total number of position degrees of freedom of all joints in the system
    int joint_coord_count = 0;

    // Returns a state object for the model
    //
    // The returned state will be initialized with the initial configuration given in
    // the model description.
    State state();

    // find potential contact pairs based on collision groups and collision mask (pairwise filtering)
    void find_shape_contact_pairs();

    // Counts the maximum number of contact points that need to be allocated.
    void count_contact_points();

    void allocate_rigid_contacts();

    // Returns a list of Tensors stored by the model
    //
    // This function is intended to be used internal-only but can be used to obtain
    // a set of all tensors owned by the model.
    void flatten();

public:
    // Maximum number of soft contacts that can be registered
    float soft_contact_max();

    // Array of per-particle radii
    float particle_radius();
};

}// namespace wp