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
    // UE5 uses cm, ALICE uses m. Convert gravity.
    Config.gravity_x = Gravity.X * 0.01;
    Config.gravity_y = Gravity.Z * 0.01; // UE5 Z-up → ALICE Y-up
    Config.gravity_z = Gravity.Y * 0.01;

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
        // ALICE (Y-up) → UE5 (Z-up)
        return FQuat(Rot.x, Rot.z, Rot.y, Rot.w);
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

FVector UAlicePhysicsWorldComponent::FromAlice(const AliceVec3& V)
{
    // ALICE (m) → UE5 (cm)
    return FVector(V.z * 100.0, V.x * 100.0, V.y * 100.0);
}
