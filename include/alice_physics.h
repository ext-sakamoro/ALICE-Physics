/**
 * ALICE-Physics: Deterministic 128-bit Fixed-Point Physics Engine
 *
 * C API Header
 *
 * Author: Moroya Sakamoto
 * License: AGPL-3.0
 */

#pragma once

#ifndef ALICE_PHYSICS_H
#define ALICE_PHYSICS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Types                                                                       */
/* ========================================================================== */

/** 3D vector (f64 at FFI boundary, Fix128 internally) */
typedef struct {
    double x;
    double y;
    double z;
} AliceVec3;

/** Quaternion rotation */
typedef struct {
    double x;
    double y;
    double z;
    double w;
} AliceQuat;

/** Physics configuration */
typedef struct {
    uint32_t substeps;
    uint32_t iterations;
    double gravity_x;
    double gravity_y;
    double gravity_z;
    double damping;
} AlicePhysicsConfig;

/** Body info snapshot (read-only) */
typedef struct {
    AliceVec3 position;
    AliceVec3 velocity;
    AliceVec3 angular_velocity;
    AliceQuat rotation;
    double inv_mass;
    uint8_t is_static;
    uint8_t is_sensor;
} AliceBodyInfo;

/** Raw Fix128 value: the solver's own 128-bit fixed-point bits (integer part hi, fraction lo). */
typedef struct {
    int64_t hi;
    uint64_t lo;
} AliceFix128Raw;

/** Raw Fix128 3-vector, for byte-for-byte determinism checks across hosts. */
typedef struct {
    AliceFix128Raw x;
    AliceFix128Raw y;
    AliceFix128Raw z;
} AliceVec3Fix128Raw;

/** Opaque physics world handle */
typedef void AlicePhysicsWorld;

/* ========================================================================== */
/* World Lifecycle                                                             */
/* ========================================================================== */

/** Create a physics world with default config. Must free with alice_physics_world_destroy. */
AlicePhysicsWorld* alice_physics_world_create(void);

/** Create a physics world with custom config. */
AlicePhysicsWorld* alice_physics_world_create_with_config(AlicePhysicsConfig config);

/** Destroy a physics world. */
void alice_physics_world_destroy(AlicePhysicsWorld* world);

/** Step the simulation by dt seconds. */
void alice_physics_world_step(AlicePhysicsWorld* world, double dt);

/** Step the simulation N times with fixed dt (batch stepping, amortizes FFI overhead). */
uint8_t alice_physics_world_step_n(AlicePhysicsWorld* world, double dt, uint32_t steps);

/** Get the number of bodies in the world. */
uint32_t alice_physics_world_body_count(const AlicePhysicsWorld* world);

/* ========================================================================== */
/* Body Management                                                             */
/* ========================================================================== */

/** Add a dynamic body. Returns body index (UINT32_MAX on error). */
uint32_t alice_physics_body_add_dynamic(AlicePhysicsWorld* world, AliceVec3 position, double mass);

/** Add a static (immovable) body. Returns body index. */
uint32_t alice_physics_body_add_static(AlicePhysicsWorld* world, AliceVec3 position);

/** Add a sensor (trigger) body. Returns body index. */
uint32_t alice_physics_body_add_sensor(AlicePhysicsWorld* world, AliceVec3 position);

/** Get body info snapshot. Returns 1 on success, 0 on failure. */
uint8_t alice_physics_body_get_info(const AlicePhysicsWorld* world, uint32_t body_id, AliceBodyInfo* out);

/** Body position as raw Fix128 bits (compare hi / lo exactly across platforms). Returns 1 on success. */
uint8_t alice_physics_body_get_position_fix128_raw(const AlicePhysicsWorld* world, uint32_t body_id, AliceVec3Fix128Raw* out);

/** Get body position. Returns 1 on success. */
uint8_t alice_physics_body_get_position(const AlicePhysicsWorld* world, uint32_t body_id, AliceVec3* out);

/** Set body position. Returns 1 on success. */
uint8_t alice_physics_body_set_position(AlicePhysicsWorld* world, uint32_t body_id, AliceVec3 position);

/** Get body velocity. Returns 1 on success. */
uint8_t alice_physics_body_get_velocity(const AlicePhysicsWorld* world, uint32_t body_id, AliceVec3* out);

/** Set body velocity. Returns 1 on success. */
uint8_t alice_physics_body_set_velocity(AlicePhysicsWorld* world, uint32_t body_id, AliceVec3 velocity);

/** Get body rotation. Returns 1 on success. */
uint8_t alice_physics_body_get_rotation(const AlicePhysicsWorld* world, uint32_t body_id, AliceQuat* out);

/** Set body restitution (bounciness, 0.0-1.0). Returns 1 on success. */
uint8_t alice_physics_body_set_restitution(AlicePhysicsWorld* world, uint32_t body_id, double restitution);

/** Set body friction coefficient. Returns 1 on success. */
uint8_t alice_physics_body_set_friction(AlicePhysicsWorld* world, uint32_t body_id, double friction);

/** Apply impulse at center of mass. Returns 1 on success. */
uint8_t alice_physics_body_apply_impulse(AlicePhysicsWorld* world, uint32_t body_id, AliceVec3 impulse);

/** Apply impulse at a world-space point. Returns 1 on success. */
uint8_t alice_physics_body_apply_impulse_at(AlicePhysicsWorld* world, uint32_t body_id, AliceVec3 impulse, AliceVec3 point);

/* ========================================================================== */
/* Batch Operations (Zero-Copy, FFI amortization)                              */
/* ========================================================================== */

/** Get all body positions as flat [x,y,z,...] f64 array. out must hold body_count*3 doubles. */
uint8_t alice_physics_world_get_positions_batch(const AlicePhysicsWorld* world, double* out, uint32_t out_capacity);

/** Get all body velocities as flat [vx,vy,vz,...] f64 array. out must hold body_count*3 doubles. */
uint8_t alice_physics_world_get_velocities_batch(const AlicePhysicsWorld* world, double* out, uint32_t out_capacity);

/** Set all body velocities from flat [vx,vy,vz,...] f64 array. count must be body_count*3. */
uint8_t alice_physics_world_set_velocities_batch(AlicePhysicsWorld* world, const double* data, uint32_t count);

/** Apply impulses in batch. data is flat [body_id, ix, iy, iz, ...]. count must be divisible by 4. */
uint8_t alice_physics_body_apply_impulses_batch(AlicePhysicsWorld* world, const double* data, uint32_t count);

/* ========================================================================== */
/* Configuration                                                               */
/* ========================================================================== */

/** Get default physics config. */
AlicePhysicsConfig alice_physics_config_default(void);

/** Set gravity on an existing world. */
void alice_physics_world_set_gravity(AlicePhysicsWorld* world, double x, double y, double z);

/** Set substeps on an existing world. */
void alice_physics_world_set_substeps(AlicePhysicsWorld* world, uint32_t substeps);

/* ========================================================================== */
/* State Serialization (Rollback Netcode)                                      */
/* ========================================================================== */

/** Serialize world state. Caller must free with alice_physics_state_free. */
uint8_t* alice_physics_state_serialize(const AlicePhysicsWorld* world, uint32_t* out_len);

/** Deserialize (restore) world state. Returns 1 on success. */
uint8_t alice_physics_state_deserialize(AlicePhysicsWorld* world, const uint8_t* data, uint32_t len);

/** Free a serialized state buffer. */
void alice_physics_state_free(uint8_t* data, uint32_t len);

/* ========================================================================== */
/* Collision radius, shapes, static colliders, joints                          */
/* ========================================================================== */

/** Collision shape. kind: 0 box (half extents a, b, c), 1 cylinder (radius a,
 *  half height b), 2 cone (radius a, half height b), 3 ellipsoid (radii a, b,
 *  c), 4 wedge (width a, height b, depth c), 5 torus (major radius a, minor
 *  radius b < a). Every used size must be finite and positive. */
typedef struct {
    uint32_t kind;
    double a;
    double b;
    double c;
} AlicePhysicsShape;

/** Set a body's collision sphere radius (finite, > 0). Returns 1 on success. */
uint8_t alice_physics_body_set_collision_radius(AlicePhysicsWorld* world, uint32_t body_id, double radius);
/** Drop a body's own collision radius (falls back to the world default). Returns 1 on success. */
uint8_t alice_physics_body_clear_collision_radius(AlicePhysicsWorld* world, uint32_t body_id);
/** Add a dynamic body with a shape; mass and inertia from density. Returns body index or UINT32_MAX. */
uint32_t alice_physics_body_add_shaped(AlicePhysicsWorld* world, AlicePhysicsShape shape, double density, AliceVec3 position);
/** Give an existing body a collision shape (mass unchanged). Returns 1 on success. */
uint8_t alice_physics_body_set_shape(AlicePhysicsWorld* world, uint32_t body_id, AlicePhysicsShape shape);

/** Add the static plane normal . p = offset. Returns static collider index or UINT32_MAX. */
uint32_t alice_physics_static_add_plane(AlicePhysicsWorld* world, AliceVec3 normal, double offset);
/** Add a static height field: width * depth heights (row-major, x fastest, both >= 2). */
uint32_t alice_physics_static_add_heightfield(AlicePhysicsWorld* world, const double* heights, uint32_t width, uint32_t depth, double spacing, AliceVec3 origin);
/** Add a static triangle mesh: vertex_count x,y,z triples and index_count indices (3 per triangle). */
uint32_t alice_physics_static_add_trimesh(AlicePhysicsWorld* world, const double* vertices, uint32_t vertex_count, const uint32_t* indices, uint32_t index_count);
/** Remove static collider index (later colliders shift down). Returns 1 on success. */
uint8_t alice_physics_static_remove(AlicePhysicsWorld* world, uint32_t index);
/** Number of static colliders. */
uint32_t alice_physics_static_count(const AlicePhysicsWorld* world);

/** Joints: anchors and axes are body-local. Each returns the joint index, or
 *  UINT32_MAX for an unknown body, body_a == body_b, or an invalid value. */
uint32_t alice_physics_joint_add_ball(AlicePhysicsWorld* world, uint32_t body_a, uint32_t body_b, AliceVec3 anchor_a, AliceVec3 anchor_b);
uint32_t alice_physics_joint_add_hinge(AlicePhysicsWorld* world, uint32_t body_a, uint32_t body_b, AliceVec3 anchor_a, AliceVec3 anchor_b, AliceVec3 axis_a, AliceVec3 axis_b);
uint32_t alice_physics_joint_add_fixed(AlicePhysicsWorld* world, uint32_t body_a, uint32_t body_b, AliceVec3 anchor_a, AliceVec3 anchor_b, AliceQuat relative_rotation);
uint32_t alice_physics_joint_add_slider(AlicePhysicsWorld* world, uint32_t body_a, uint32_t body_b, AliceVec3 axis, AliceVec3 anchor_a, AliceVec3 anchor_b);
uint32_t alice_physics_joint_add_spring(AlicePhysicsWorld* world, uint32_t body_a, uint32_t body_b, AliceVec3 anchor_a, AliceVec3 anchor_b, double rest_length, double stiffness, double damping);
/** Remove joint index; the last joint moves into index. Returns 1 on success. */
uint8_t alice_physics_joint_remove(AlicePhysicsWorld* world, uint32_t index);
/** Number of joints. */
uint32_t alice_physics_joint_count(const AlicePhysicsWorld* world);

/* ========================================================================== */
/* World queries and body observation                                          */
/* ========================================================================== */

/** exclude_body value meaning "exclude no body"; AliceQueryHit.body value
 *  meaning "the hit belongs to no body". */
#define ALICE_PHYSICS_NO_BODY UINT32_MAX

/** Hit of a query against the collided geometry (bodies and their shapes,
 *  static colliders, SDF colliders). target_kind: 0 body, 1 static collider,
 *  2 SDF collider. */
typedef struct {
    double t;
    AliceVec3 point;
    AliceVec3 normal;
    uint32_t target_kind;
    uint32_t target_index;
    uint32_t body;
} AliceQueryHit;

/** One collider found by an overlap query (kind as AliceQueryHit.target_kind). */
typedef struct {
    uint32_t kind;
    uint32_t index;
} AliceQueryTarget;

/** Observation of one body. */
typedef struct {
    uint32_t body_index;
    AliceVec3 position;
    AliceVec3 velocity;
    AliceQuat rotation;
    AliceVec3 angular_velocity;
    uint8_t sleeping;
    uint8_t in_contact;
} AliceBodyObservation;

/** Nearest ray hit. Returns 1 and writes out on a hit; 0 for no hit (also a
 *  zero direction or max_t <= 0), a null world / out, a non-finite value or an
 *  exclude_body that is not a body (out unchanged). */
uint8_t alice_physics_world_cast_ray(const AlicePhysicsWorld* world, AliceVec3 origin, AliceVec3 direction, double max_t, uint32_t exclude_body, AliceQueryHit* out);
/** Nearest hit of a sphere moving along direction (negative radius: no hit). Returns as cast_ray. */
uint8_t alice_physics_world_cast_sphere(const AlicePhysicsWorld* world, AliceVec3 center, double radius, AliceVec3 direction, double max_t, uint32_t exclude_body, AliceQueryHit* out);
/** Nearest hit of a capsule (segment a-b grown by radius) moving along direction. Returns as cast_ray. */
uint8_t alice_physics_world_cast_capsule(const AlicePhysicsWorld* world, AliceVec3 a, AliceVec3 b, double radius, AliceVec3 direction, double max_t, uint32_t exclude_body, AliceQueryHit* out);
/** Colliders a sphere overlaps, sorted by kind then index. Returns the number
 *  found and writes the first min(found, capacity) to out (out may be NULL
 *  when capacity is 0); a return value above capacity means call again with a
 *  larger buffer. UINT32_MAX for a null world, NULL out with capacity > 0, a
 *  non-finite value or an exclude_body that is not a body. */
uint32_t alice_physics_world_overlap_sphere(const AlicePhysicsWorld* world, AliceVec3 center, double radius, uint32_t exclude_body, AliceQueryTarget* out, uint32_t capacity);
/** Observe one body. Returns 1 on success, 0 for a null world / out or an unknown body. */
uint8_t alice_physics_body_observe(const AlicePhysicsWorld* world, uint32_t body_id, AliceBodyObservation* out);

/* ========================================================================== */
/* Version                                                                     */
/* ========================================================================== */

/** Get library version string (null-terminated). */
const char* alice_physics_version(void);

/* ---- Panic isolation (1.2.0) ------------------------------------------------
 * Every function above catches Rust panics inside the call and returns its
 * documented sentinel (0 / UINT32_MAX / NULL) instead of aborting the host.
 * The panic message is kept per thread: */

/** Most recent panic message on this thread (heap string, take semantics: a
 *  second call returns NULL until the next error), or NULL. Free with
 *  alice_physics_string_free. */
char* alice_physics_last_error(void);

/** Discard the most recent panic message on this thread. */
void alice_physics_clear_last_error(void);

/** Free a string returned by alice_physics_last_error (NULL is a no-op). */
void alice_physics_string_free(char* s);

#ifdef __cplusplus
}
#endif

#endif /* ALICE_PHYSICS_H */
