// ALICE-Physics UE5 Component Implementation
// Author: Moroya Sakamoto

#include "AlicePhysicsComponent.h"

UAlicePhysicsWorldComponent::UAlicePhysicsWorldComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
}

void UAlicePhysicsWorldComponent::BeginPlay()
{
    Super::BeginPlay();

    AlicePhysicsConfig Config = alice_physics_config_default();
    Config.substeps = static_cast<uint32_t>(Substeps);
    // Same axis mapping and cm -> m scale as positions (ToAlice).
    const AliceVec3 G = ToAlice(Gravity);
    Config.gravity_x = G.x;
    Config.gravity_y = G.y;
    Config.gravity_z = G.z;

    World = reinterpret_cast<AlicePhysicsWorld*>(
        alice_physics_world_create_with_config(Config));
}

void UAlicePhysicsWorldComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (World)
    {
        alice_physics_world_destroy(World);
        World = nullptr;
    }
    Super::EndPlay(EndPlayReason);
}

void UAlicePhysicsWorldComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (bAutoStep && World)
    {
        alice_physics_world_step(World, static_cast<double>(DeltaTime));
    }
}

// -- Body Management --

int32 UAlicePhysicsWorldComponent::AddDynamicBody(FVector Position, float Mass)
{
    if (!World) return -1;
    return static_cast<int32>(
        alice_physics_body_add_dynamic(World, ToAlice(Position), static_cast<double>(Mass)));
}

int32 UAlicePhysicsWorldComponent::AddStaticBody(FVector Position)
{
    if (!World) return -1;
    return static_cast<int32>(
        alice_physics_body_add_static(World, ToAlice(Position)));
}

FVector UAlicePhysicsWorldComponent::GetBodyPosition(int32 BodyId) const
{
    if (!World) return FVector::ZeroVector;
    AliceVec3 Pos;
    if (alice_physics_body_get_position(World, static_cast<uint32_t>(BodyId), &Pos))
    {
        return FromAlice(Pos);
    }
    return FVector::ZeroVector;
}

FQuat UAlicePhysicsWorldComponent::GetBodyRotation(int32 BodyId) const
{
    if (!World) return FQuat::Identity;
    AliceQuat Rot;
    if (alice_physics_body_get_rotation(World, static_cast<uint32_t>(BodyId), &Rot))
    {
        return QuatFromAlice(Rot);
    }
    return FQuat::Identity;
}

FVector UAlicePhysicsWorldComponent::GetBodyVelocity(int32 BodyId) const
{
    if (!World) return FVector::ZeroVector;
    AliceVec3 Vel;
    if (alice_physics_body_get_velocity(World, static_cast<uint32_t>(BodyId), &Vel))
    {
        return FromAlice(Vel);
    }
    return FVector::ZeroVector;
}

void UAlicePhysicsWorldComponent::ApplyImpulse(int32 BodyId, FVector Impulse)
{
    if (!World) return;
    alice_physics_body_apply_impulse(World, static_cast<uint32_t>(BodyId), ToAlice(Impulse));
}

void UAlicePhysicsWorldComponent::ApplyImpulseAt(int32 BodyId, FVector Impulse, FVector Point)
{
    if (!World) return;
    alice_physics_body_apply_impulse_at(World, static_cast<uint32_t>(BodyId), ToAlice(Impulse), ToAlice(Point));
}

// -- Simulation --

void UAlicePhysicsWorldComponent::StepSimulation(float DeltaTime)
{
    if (World)
    {
        alice_physics_world_step(World, static_cast<double>(DeltaTime));
    }
}

int32 UAlicePhysicsWorldComponent::GetBodyCount() const
{
    if (!World) return 0;
    return static_cast<int32>(alice_physics_world_body_count(World));
}

int32 UAlicePhysicsWorldComponent::AddSensorBody(FVector Position)
{
    if (!World) return -1;
    return IndexOrMinusOne(alice_physics_body_add_sensor(World, ToAlice(Position)));
}

bool UAlicePhysicsWorldComponent::SetBodyPosition(int32 BodyId, FVector Position)
{
    if (!World || BodyId < 0) return false;
    return alice_physics_body_set_position(World, static_cast<uint32_t>(BodyId), ToAlice(Position)) != 0;
}

bool UAlicePhysicsWorldComponent::SetBodyVelocity(int32 BodyId, FVector Velocity)
{
    if (!World || BodyId < 0) return false;
    return alice_physics_body_set_velocity(World, static_cast<uint32_t>(BodyId), ToAlice(Velocity)) != 0;
}

bool UAlicePhysicsWorldComponent::SetBodyRestitution(int32 BodyId, float Restitution)
{
    if (!World || BodyId < 0) return false;
    return alice_physics_body_set_restitution(World, static_cast<uint32_t>(BodyId), static_cast<double>(Restitution)) != 0;
}

bool UAlicePhysicsWorldComponent::SetBodyFriction(int32 BodyId, float Friction)
{
    if (!World || BodyId < 0) return false;
    return alice_physics_body_set_friction(World, static_cast<uint32_t>(BodyId), static_cast<double>(Friction)) != 0;
}

// -- Collision Radius and Shapes --

bool UAlicePhysicsWorldComponent::SetCollisionRadius(int32 BodyId, float RadiusCm)
{
    if (!World || BodyId < 0) return false;
    return alice_physics_body_set_collision_radius(World, static_cast<uint32_t>(BodyId), RadiusCm * 0.01) != 0;
}

bool UAlicePhysicsWorldComponent::ClearCollisionRadius(int32 BodyId)
{
    if (!World || BodyId < 0) return false;
    return alice_physics_body_clear_collision_radius(World, static_cast<uint32_t>(BodyId)) != 0;
}

int32 UAlicePhysicsWorldComponent::AddShapedBody(int32 Kind, FVector SizeCm, float Density, FVector Position)
{
    if (!World) return -1;
    return IndexOrMinusOne(alice_physics_body_add_shaped(
        World, ShapeToAlice(Kind, SizeCm), static_cast<double>(Density), ToAlice(Position)));
}

bool UAlicePhysicsWorldComponent::SetBodyShape(int32 BodyId, int32 Kind, FVector SizeCm)
{
    if (!World || BodyId < 0) return false;
    return alice_physics_body_set_shape(World, static_cast<uint32_t>(BodyId), ShapeToAlice(Kind, SizeCm)) != 0;
}

// -- Static Colliders --

int32 UAlicePhysicsWorldComponent::AddStaticPlane(FVector Normal, FVector Point)
{
    if (!World) return -1;
    // normal . p = offset with a unit normal; the C ABI normalises, so take
    // the offset along the normalised direction.
    const FVector N = Normal.GetSafeNormal();
    if (N.IsZero()) return -1;
    const AliceVec3 NA = DirToAlice(N);
    const AliceVec3 PA = ToAlice(Point);
    const double Offset = NA.x * PA.x + NA.y * PA.y + NA.z * PA.z;
    return IndexOrMinusOne(alice_physics_static_add_plane(World, NA, Offset));
}

int32 UAlicePhysicsWorldComponent::AddStaticHeightField(const TArray<float>& HeightsCm, int32 Width, int32 Depth, float SpacingCm, FVector OriginCm)
{
    if (!World || Width < 2 || Depth < 2 || HeightsCm.Num() != Width * Depth) return -1;
    // ALICE's grid runs along its X (= UE Y) and Z (= UE X), x fastest, which
    // is the I / J order documented on the declaration; heights are UE Z = ALICE Y.
    TArray<double> Heights;
    Heights.SetNumUninitialized(HeightsCm.Num());
    for (int32 K = 0; K < HeightsCm.Num(); ++K)
    {
        Heights[K] = HeightsCm[K] * 0.01;
    }
    AliceVec3 Origin = ToAlice(OriginCm);
    Origin.y = 0.0; // heights are absolute
    return IndexOrMinusOne(alice_physics_static_add_heightfield(
        World, Heights.GetData(), static_cast<uint32_t>(Width), static_cast<uint32_t>(Depth),
        SpacingCm * 0.01, Origin));
}

int32 UAlicePhysicsWorldComponent::AddStaticTriMesh(const TArray<FVector>& Vertices, const TArray<int32>& Indices)
{
    if (!World || Vertices.Num() == 0 || Indices.Num() == 0) return -1;
    TArray<double> Coords;
    Coords.SetNumUninitialized(Vertices.Num() * 3);
    for (int32 K = 0; K < Vertices.Num(); ++K)
    {
        const AliceVec3 V = ToAlice(Vertices[K]);
        Coords[3 * K] = V.x;
        Coords[3 * K + 1] = V.y;
        Coords[3 * K + 2] = V.z;
    }
    TArray<uint32_t> Idx;
    Idx.SetNumUninitialized(Indices.Num());
    for (int32 K = 0; K < Indices.Num(); ++K)
    {
        if (Indices[K] < 0) return -1;
        Idx[K] = static_cast<uint32_t>(Indices[K]);
    }
    // The axis permutation of ToAlice is cyclic (a rotation), so triangle
    // winding is unchanged.
    return IndexOrMinusOne(alice_physics_static_add_trimesh(
        World, Coords.GetData(), static_cast<uint32_t>(Vertices.Num()), Idx.GetData(), static_cast<uint32_t>(Idx.Num())));
}

bool UAlicePhysicsWorldComponent::RemoveStaticCollider(int32 Index)
{
    if (!World || Index < 0) return false;
    return alice_physics_static_remove(World, static_cast<uint32_t>(Index)) != 0;
}

int32 UAlicePhysicsWorldComponent::GetStaticColliderCount() const
{
    if (!World) return 0;
    return static_cast<int32>(alice_physics_static_count(World));
}

// -- Joints --

int32 UAlicePhysicsWorldComponent::AddBallJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB)
{
    if (!World || BodyA < 0 || BodyB < 0) return -1;
    return IndexOrMinusOne(alice_physics_joint_add_ball(
        World, static_cast<uint32_t>(BodyA), static_cast<uint32_t>(BodyB), ToAlice(AnchorA), ToAlice(AnchorB)));
}

int32 UAlicePhysicsWorldComponent::AddHingeJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB, FVector AxisA, FVector AxisB)
{
    if (!World || BodyA < 0 || BodyB < 0) return -1;
    return IndexOrMinusOne(alice_physics_joint_add_hinge(
        World, static_cast<uint32_t>(BodyA), static_cast<uint32_t>(BodyB), ToAlice(AnchorA), ToAlice(AnchorB),
        DirToAlice(AxisA), DirToAlice(AxisB)));
}

int32 UAlicePhysicsWorldComponent::AddFixedJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB, FQuat RelativeRotation)
{
    if (!World || BodyA < 0 || BodyB < 0) return -1;
    return IndexOrMinusOne(alice_physics_joint_add_fixed(
        World, static_cast<uint32_t>(BodyA), static_cast<uint32_t>(BodyB), ToAlice(AnchorA), ToAlice(AnchorB),
        QuatToAlice(RelativeRotation)));
}

int32 UAlicePhysicsWorldComponent::AddSliderJoint(int32 BodyA, int32 BodyB, FVector Axis, FVector AnchorA, FVector AnchorB)
{
    if (!World || BodyA < 0 || BodyB < 0) return -1;
    return IndexOrMinusOne(alice_physics_joint_add_slider(
        World, static_cast<uint32_t>(BodyA), static_cast<uint32_t>(BodyB), DirToAlice(Axis), ToAlice(AnchorA), ToAlice(AnchorB)));
}

int32 UAlicePhysicsWorldComponent::AddSpringJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB, float RestLengthCm, float Stiffness, float Damping)
{
    if (!World || BodyA < 0 || BodyB < 0) return -1;
    return IndexOrMinusOne(alice_physics_joint_add_spring(
        World, static_cast<uint32_t>(BodyA), static_cast<uint32_t>(BodyB), ToAlice(AnchorA), ToAlice(AnchorB),
        RestLengthCm * 0.01, static_cast<double>(Stiffness), static_cast<double>(Damping)));
}

bool UAlicePhysicsWorldComponent::RemoveJoint(int32 Index)
{
    if (!World || Index < 0) return false;
    return alice_physics_joint_remove(World, static_cast<uint32_t>(Index)) != 0;
}

int32 UAlicePhysicsWorldComponent::GetJointCount() const
{
    if (!World) return 0;
    return static_cast<int32>(alice_physics_joint_count(World));
}

// -- Simulation (continued) --

void UAlicePhysicsWorldComponent::StepSimulationN(float DeltaTime, int32 Steps)
{
    if (World && Steps > 0)
    {
        alice_physics_world_step_n(World, static_cast<double>(DeltaTime), static_cast<uint32_t>(Steps));
    }
}

bool UAlicePhysicsWorldComponent::SetWorldGravity(FVector NewGravity)
{
    if (!World) return false;
    const AliceVec3 G = ToAlice(NewGravity);
    alice_physics_world_set_gravity(World, G.x, G.y, G.z);
    Gravity = NewGravity;
    return true;
}

bool UAlicePhysicsWorldComponent::SetWorldSubsteps(int32 NewSubsteps)
{
    if (!World || NewSubsteps < 1) return false;
    alice_physics_world_set_substeps(World, static_cast<uint32_t>(NewSubsteps));
    Substeps = NewSubsteps;
    return true;
}

// -- Diagnostics --

FString UAlicePhysicsWorldComponent::TakeLastError()
{
    char* Msg = alice_physics_last_error();
    if (!Msg) return FString();
    FString Result = UTF8_TO_TCHAR(Msg);
    alice_physics_string_free(Msg);
    return Result;
}

void UAlicePhysicsWorldComponent::ClearLastError()
{
    alice_physics_clear_last_error();
}

FString UAlicePhysicsWorldComponent::GetLibraryVersion()
{
    const char* V = alice_physics_version();
    return V ? FString(UTF8_TO_TCHAR(V)) : FString();
}

// -- State Serialization --

TArray<uint8> UAlicePhysicsWorldComponent::SerializeState() const
{
    TArray<uint8> Result;
    if (!World) return Result;

    uint32_t Len = 0;
    uint8_t* Data = alice_physics_state_serialize(World, &Len);
    if (Data && Len > 0)
    {
        Result.SetNumUninitialized(Len);
        FMemory::Memcpy(Result.GetData(), Data, Len);
        alice_physics_state_free(Data, Len);
    }
    return Result;
}

bool UAlicePhysicsWorldComponent::DeserializeState(const TArray<uint8>& Data)
{
    if (!World || Data.Num() == 0) return false;
    return alice_physics_state_deserialize(World, Data.GetData(), static_cast<uint32_t>(Data.Num())) != 0;
}

// -- Helpers --

AliceVec3 UAlicePhysicsWorldComponent::ToAlice(const FVector& V)
{
    // UE5 (X-forward, Y-right, Z-up, cm) → ALICE (X-right, Y-up, Z-forward, m)
    AliceVec3 Result;
    Result.x = V.Y * 0.01;
    Result.y = V.Z * 0.01;
    Result.z = V.X * 0.01;
    return Result;
}

AliceVec3 UAlicePhysicsWorldComponent::DirToAlice(const FVector& V)
{
    AliceVec3 Result;
    Result.x = V.Y;
    Result.y = V.Z;
    Result.z = V.X;
    return Result;
}

AliceQuat UAlicePhysicsWorldComponent::QuatToAlice(const FQuat& Q)
{
    // ToAlice permutes the axes cyclically (a proper rotation of the frame),
    // so a quaternion's vector part permutes like a direction and w is kept.
    AliceQuat Result;
    Result.x = Q.Y;
    Result.y = Q.Z;
    Result.z = Q.X;
    Result.w = Q.W;
    return Result;
}

FQuat UAlicePhysicsWorldComponent::QuatFromAlice(const AliceQuat& Q)
{
    // Inverse of QuatToAlice: UE (X, Y, Z) = ALICE (z, x, y), w kept.
    return FQuat(Q.z, Q.x, Q.y, Q.w);
}

AlicePhysicsShape UAlicePhysicsWorldComponent::ShapeToAlice(int32 Kind, const FVector& SizeCm)
{
    AlicePhysicsShape S;
    S.kind = Kind < 0 ? UINT32_MAX : static_cast<uint32_t>(Kind);
    const FVector M = SizeCm * 0.01;
    switch (Kind)
    {
    case 0: // box half extents
    case 3: // ellipsoid radii
        S.a = M.Y; S.b = M.Z; S.c = M.X;
        break;
    case 1: // cylinder
    case 2: // cone: radius, half height along the up axis
        S.a = M.X; S.b = M.Z; S.c = 0.0;
        break;
    case 4: // wedge: width (UE Y), height (UE Z), depth (UE X)
        S.a = M.Y; S.b = M.Z; S.c = M.X;
        break;
    case 5: // torus: major, minor
        S.a = M.X; S.b = M.Y; S.c = 0.0;
        break;
    default: // refused by the library
        S.a = 0.0; S.b = 0.0; S.c = 0.0;
        break;
    }
    return S;
}

int32 UAlicePhysicsWorldComponent::IndexOrMinusOne(uint32_t Index)
{
    return Index == UINT32_MAX || Index > static_cast<uint32_t>(INT32_MAX) ? -1 : static_cast<int32>(Index);
}

FVector UAlicePhysicsWorldComponent::FromAlice(const AliceVec3& V)
{
    // ALICE (m) → UE5 (cm)
    return FVector(V.z * 100.0, V.x * 100.0, V.y * 100.0);
}
