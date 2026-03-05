// ALICE-Physics UE5 C++ Header
// 30 FFI functions for deterministic 128-bit fixed-point physics
//
// Author: Moroya Sakamoto

#pragma once

#include <cstdint>
#include <cstring>
#include <utility>

// ============================================================================
// C API
// ============================================================================

extern "C" {

// Opaque handle
typedef void* AlicePhysicsWorldHandle;

// C-compatible 3D vector (f64 for FFI boundary)
struct AliceVec3 {
    double x;
    double y;
    double z;
};

// C-compatible quaternion
struct AliceQuat {
    double x;
    double y;
    double z;
    double w;
};

// C-compatible physics config
struct AlicePhysicsConfig {
    uint32_t substeps;
    uint32_t iterations;
    double gravity_x;
    double gravity_y;
    double gravity_z;
    double damping;
};

// C-compatible body info (read-only snapshot)
struct AliceBodyInfo {
    AliceVec3 position;
    AliceVec3 velocity;
    AliceVec3 angular_velocity;
    AliceQuat rotation;
    double inv_mass;
    uint8_t is_static;
    uint8_t is_sensor;
};

// --- World lifecycle ---
AlicePhysicsWorldHandle alice_physics_world_create();
AlicePhysicsWorldHandle alice_physics_world_create_with_config(AlicePhysicsConfig config);
void     alice_physics_world_destroy(AlicePhysicsWorldHandle world);

// --- Simulation ---
uint8_t  alice_physics_world_step(AlicePhysicsWorldHandle world, double dt);
uint8_t  alice_physics_world_step_n(AlicePhysicsWorldHandle world, double dt, uint32_t steps);
uint32_t alice_physics_world_body_count(const AlicePhysicsWorldHandle world);

// --- Batch accessors ---
uint8_t  alice_physics_world_get_positions_batch(const AlicePhysicsWorldHandle world, double* out, uint32_t out_capacity);
uint8_t  alice_physics_world_get_velocities_batch(const AlicePhysicsWorldHandle world, double* out, uint32_t out_capacity);
uint8_t  alice_physics_world_set_velocities_batch(AlicePhysicsWorldHandle world, const double* data, uint32_t count);
uint8_t  alice_physics_body_apply_impulses_batch(AlicePhysicsWorldHandle world, const double* data, uint32_t count);

// --- Body management ---
uint32_t alice_physics_body_add_dynamic(AlicePhysicsWorldHandle world, AliceVec3 position, double mass);
uint32_t alice_physics_body_add_static(AlicePhysicsWorldHandle world, AliceVec3 position);
uint32_t alice_physics_body_add_sensor(AlicePhysicsWorldHandle world, AliceVec3 position);

// --- Body accessors ---
uint8_t  alice_physics_body_get_info(const AlicePhysicsWorldHandle world, uint32_t body_id, AliceBodyInfo* out);
uint8_t  alice_physics_body_get_position(const AlicePhysicsWorldHandle world, uint32_t body_id, AliceVec3* out);
uint8_t  alice_physics_body_set_position(AlicePhysicsWorldHandle world, uint32_t body_id, AliceVec3 position);
uint8_t  alice_physics_body_get_velocity(const AlicePhysicsWorldHandle world, uint32_t body_id, AliceVec3* out);
uint8_t  alice_physics_body_set_velocity(AlicePhysicsWorldHandle world, uint32_t body_id, AliceVec3 velocity);
uint8_t  alice_physics_body_get_rotation(const AlicePhysicsWorldHandle world, uint32_t body_id, AliceQuat* out);
uint8_t  alice_physics_body_set_restitution(AlicePhysicsWorldHandle world, uint32_t body_id, double restitution);
uint8_t  alice_physics_body_set_friction(AlicePhysicsWorldHandle world, uint32_t body_id, double friction);

// --- Impulses ---
uint8_t  alice_physics_body_apply_impulse(AlicePhysicsWorldHandle world, uint32_t body_id, AliceVec3 impulse);
uint8_t  alice_physics_body_apply_impulse_at(AlicePhysicsWorldHandle world, uint32_t body_id, AliceVec3 impulse, AliceVec3 point);

// --- Config ---
AlicePhysicsConfig alice_physics_config_default();
void     alice_physics_world_set_gravity(AlicePhysicsWorldHandle world, double x, double y, double z);
void     alice_physics_world_set_substeps(AlicePhysicsWorldHandle world, uint32_t substeps);

// --- State serialization (rollback netcode) ---
uint8_t* alice_physics_state_serialize(const AlicePhysicsWorldHandle world, uint32_t* out_len);
uint8_t  alice_physics_state_deserialize(AlicePhysicsWorldHandle world, const uint8_t* data, uint32_t len);
void     alice_physics_state_free(uint8_t* data, uint32_t len);

// --- Version ---
const char* alice_physics_version();

} // extern "C"

// ============================================================================
// RAII C++ Wrapper
// ============================================================================

namespace AlicePhysics {

/// RAII wrapper for the deterministic physics world
class FPhysicsWorld {
public:
    FPhysicsWorld()
        : Handle(alice_physics_world_create()) {}

    explicit FPhysicsWorld(AlicePhysicsConfig Config)
        : Handle(alice_physics_world_create_with_config(Config)) {}

    ~FPhysicsWorld() {
        if (Handle) alice_physics_world_destroy(Handle);
    }

    // Move only
    FPhysicsWorld(FPhysicsWorld&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FPhysicsWorld& operator=(FPhysicsWorld&& Other) noexcept {
        if (this != &Other) {
            if (Handle) alice_physics_world_destroy(Handle);
            Handle = Other.Handle;
            Other.Handle = nullptr;
        }
        return *this;
    }
    FPhysicsWorld(const FPhysicsWorld&) = delete;
    FPhysicsWorld& operator=(const FPhysicsWorld&) = delete;

    // Simulation
    bool Step(double Dt) { return alice_physics_world_step(Handle, Dt) != 0; }
    bool StepN(double Dt, uint32_t Steps) { return alice_physics_world_step_n(Handle, Dt, Steps) != 0; }
    uint32_t BodyCount() const { return alice_physics_world_body_count(Handle); }

    // Body management
    uint32_t AddDynamic(AliceVec3 Pos, double Mass) { return alice_physics_body_add_dynamic(Handle, Pos, Mass); }
    uint32_t AddStatic(AliceVec3 Pos) { return alice_physics_body_add_static(Handle, Pos); }
    uint32_t AddSensor(AliceVec3 Pos) { return alice_physics_body_add_sensor(Handle, Pos); }

    // Body accessors
    bool GetBodyInfo(uint32_t Id, AliceBodyInfo& Out) const { return alice_physics_body_get_info(Handle, Id, &Out) != 0; }
    bool GetPosition(uint32_t Id, AliceVec3& Out) const { return alice_physics_body_get_position(Handle, Id, &Out) != 0; }
    bool SetPosition(uint32_t Id, AliceVec3 Pos) { return alice_physics_body_set_position(Handle, Id, Pos) != 0; }
    bool GetVelocity(uint32_t Id, AliceVec3& Out) const { return alice_physics_body_get_velocity(Handle, Id, &Out) != 0; }
    bool SetVelocity(uint32_t Id, AliceVec3 Vel) { return alice_physics_body_set_velocity(Handle, Id, Vel) != 0; }
    bool GetRotation(uint32_t Id, AliceQuat& Out) const { return alice_physics_body_get_rotation(Handle, Id, &Out) != 0; }
    bool SetRestitution(uint32_t Id, double R) { return alice_physics_body_set_restitution(Handle, Id, R) != 0; }
    bool SetFriction(uint32_t Id, double F) { return alice_physics_body_set_friction(Handle, Id, F) != 0; }

    // Impulses
    bool ApplyImpulse(uint32_t Id, AliceVec3 Imp) { return alice_physics_body_apply_impulse(Handle, Id, Imp) != 0; }
    bool ApplyImpulseAt(uint32_t Id, AliceVec3 Imp, AliceVec3 Pt) { return alice_physics_body_apply_impulse_at(Handle, Id, Imp, Pt) != 0; }

    // Batch
    bool GetPositionsBatch(double* Out, uint32_t Cap) const { return alice_physics_world_get_positions_batch(Handle, Out, Cap) != 0; }
    bool GetVelocitiesBatch(double* Out, uint32_t Cap) const { return alice_physics_world_get_velocities_batch(Handle, Out, Cap) != 0; }
    bool SetVelocitiesBatch(const double* Data, uint32_t Count) { return alice_physics_world_set_velocities_batch(Handle, Data, Count) != 0; }
    bool ApplyImpulsesBatch(const double* Data, uint32_t Count) { return alice_physics_body_apply_impulses_batch(Handle, Data, Count) != 0; }

    // Config
    void SetGravity(double X, double Y, double Z) { alice_physics_world_set_gravity(Handle, X, Y, Z); }
    void SetSubsteps(uint32_t N) { alice_physics_world_set_substeps(Handle, N); }

    // State serialization
    struct SerializedState {
        uint8_t* Data = nullptr;
        uint32_t Len = 0;
        ~SerializedState() { if (Data) alice_physics_state_free(Data, Len); }
        SerializedState() = default;
        SerializedState(SerializedState&& O) noexcept : Data(O.Data), Len(O.Len) { O.Data = nullptr; O.Len = 0; }
        SerializedState& operator=(SerializedState&& O) noexcept {
            if (this != &O) { if (Data) alice_physics_state_free(Data, Len); Data = O.Data; Len = O.Len; O.Data = nullptr; O.Len = 0; }
            return *this;
        }
        SerializedState(const SerializedState&) = delete;
        SerializedState& operator=(const SerializedState&) = delete;
    };

    SerializedState Serialize() const {
        SerializedState S;
        S.Data = alice_physics_state_serialize(Handle, &S.Len);
        return S;
    }

    bool Deserialize(const uint8_t* Data, uint32_t Len) {
        return alice_physics_state_deserialize(Handle, Data, Len) != 0;
    }

    bool IsValid() const { return Handle != nullptr; }

private:
    AlicePhysicsWorldHandle Handle = nullptr;
};

} // namespace AlicePhysics
