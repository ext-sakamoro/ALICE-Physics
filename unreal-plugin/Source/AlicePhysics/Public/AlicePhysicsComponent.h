// ALICE-Physics UE5 Component
// Author: Moroya Sakamoto

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "alice_physics.h"
#include "AlicePhysicsComponent.generated.h"

/**
 * UAlicePhysicsWorldComponent
 *
 * Manages an ALICE-Physics deterministic simulation world.
 * Attach to an actor to create a physics world, add bodies, and step the simulation.
 *
 * Designed for rollback netcode multiplayer games requiring bit-exact determinism.
 */
UCLASS(ClassGroup=(Physics), meta=(BlueprintSpawnableComponent))
class ALICEPHYSICS_API UAlicePhysicsWorldComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UAlicePhysicsWorldComponent();

    // -- Lifecycle --

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

    // -- World Config --

    /** Number of substeps per frame */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE Physics")
    int32 Substeps = 8;

    /** Gravity vector */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE Physics")
    FVector Gravity = FVector(0, 0, -981.0);

    /** Whether to auto-step each tick */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE Physics")
    bool bAutoStep = true;

    // -- Body Management --

    /** Add a dynamic body. Returns body ID. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    int32 AddDynamicBody(FVector Position, float Mass);

    /** Add a static body. Returns body ID. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    int32 AddStaticBody(FVector Position);

    /** Get body position. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    FVector GetBodyPosition(int32 BodyId) const;

    /** Get body rotation. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    FQuat GetBodyRotation(int32 BodyId) const;

    /** Get body velocity. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    FVector GetBodyVelocity(int32 BodyId) const;

    /** Apply impulse at center of mass. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    void ApplyImpulse(int32 BodyId, FVector Impulse);

    /** Apply impulse at a world-space point. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    void ApplyImpulseAt(int32 BodyId, FVector Impulse, FVector Point);

    // -- Simulation --

    /** Step simulation manually. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    void StepSimulation(float DeltaTime);

    /** Get number of bodies. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    int32 GetBodyCount() const;

    // -- State Serialization --

    /** Serialize world state (for rollback). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Netcode")
    TArray<uint8> SerializeState() const;

    /** Deserialize world state (rollback restore). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Netcode")
    bool DeserializeState(const TArray<uint8>& Data);

private:
    AlicePhysicsWorld* World = nullptr;

    static AliceVec3 ToAlice(const FVector& V);
    static FVector FromAlice(const AliceVec3& V);
};
