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
/* Version                                                                     */
/* ========================================================================== */

/** Get library version string (null-terminated). */
const char* alice_physics_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ALICE_PHYSICS_H */
