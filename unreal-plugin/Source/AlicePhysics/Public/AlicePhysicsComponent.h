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

    /** Add a sensor (trigger) body. Returns body ID. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    int32 AddSensorBody(FVector Position);

    /** Set body position. Returns false for an unknown body. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    bool SetBodyPosition(int32 BodyId, FVector Position);

    /** Set body velocity. Returns false for an unknown body. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    bool SetBodyVelocity(int32 BodyId, FVector Velocity);

    /** Set body restitution (0-1). Returns false for an unknown body. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    bool SetBodyRestitution(int32 BodyId, float Restitution);

    /** Set body friction coefficient. Returns false for an unknown body. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    bool SetBodyFriction(int32 BodyId, float Friction);

    // -- Collision Radius and Shapes --

    /** Set a body's collision sphere radius (cm, > 0). Returns false when refused. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Shapes")
    bool SetCollisionRadius(int32 BodyId, float RadiusCm);

    /** Drop a body's own collision radius (it falls back to the world default). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Shapes")
    bool ClearCollisionRadius(int32 BodyId);

    /**
     * Add a dynamic body with a collision shape; mass and inertia come from
     * Density (kg/m^3). Kind: 0 box, 1 cylinder, 2 cone, 3 ellipsoid, 4 wedge,
     * 5 torus. SizeCm (UE axes, cm): box = half extents, ellipsoid = radii,
     * cylinder / cone = (radius, unused, half height along Z), wedge =
     * (depth, width, height), torus = (major radius, minor radius, unused).
     * Returns body ID or -1 when refused.
     */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Shapes")
    int32 AddShapedBody(int32 Kind, FVector SizeCm, float Density, FVector Position);

    /** Give an existing body a collision shape (same Kind / SizeCm as AddShapedBody). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Shapes")
    bool SetBodyShape(int32 BodyId, int32 Kind, FVector SizeCm);

    // -- Static Colliders --

    /** Add the static plane through Point (cm) with the given normal. Returns its index or -1. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Static")
    int32 AddStaticPlane(FVector Normal, FVector Point);

    /**
     * Add a static height field. HeightsCm[I + J * Width] is the surface
     * height (UE Z, cm) at OriginCm + (J * SpacingCm, I * SpacingCm, 0): I runs
     * along UE +Y (Width points), J along UE +X (Depth points), both >= 2.
     * Returns its index or -1.
     */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Static")
    int32 AddStaticHeightField(const TArray<float>& HeightsCm, int32 Width, int32 Depth, float SpacingCm, FVector OriginCm);

    /** Add a static triangle mesh (vertices in cm, three indices per triangle). Returns its index or -1. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Static")
    int32 AddStaticTriMesh(const TArray<FVector>& Vertices, const TArray<int32>& Indices);

    /** Remove static collider Index (later colliders shift down by one). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Static")
    bool RemoveStaticCollider(int32 Index);

    /** Number of static colliders. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Static")
    int32 GetStaticColliderCount() const;

    // -- Joints (anchors in body-local cm, axes body-local) --

    /** Ball-and-socket joint. Returns joint index or -1 (unknown body, BodyA == BodyB). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Joints")
    int32 AddBallJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB);

    /** Hinge joint: the anchors meet and AxisA stays aligned with AxisB. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Joints")
    int32 AddHingeJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB, FVector AxisA, FVector AxisB);

    /** Fixed joint: BodyB keeps RelativeRotation relative to BodyA. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Joints")
    int32 AddFixedJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB, FQuat RelativeRotation);

    /** Slider joint along Axis (BodyA local). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Joints")
    int32 AddSliderJoint(int32 BodyA, int32 BodyB, FVector Axis, FVector AnchorA, FVector AnchorB);

    /** Spring between the anchors (rest length in cm, stiffness N/m, damping N*s/m). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Joints")
    int32 AddSpringJoint(int32 BodyA, int32 BodyB, FVector AnchorA, FVector AnchorB, float RestLengthCm, float Stiffness, float Damping);

    /** Remove joint Index (the last joint moves into Index). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Joints")
    bool RemoveJoint(int32 Index);

    /** Number of joints. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Joints")
    int32 GetJointCount() const;

    // -- Simulation (continued) --

    /** Step Steps times by DeltaTime. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    void StepSimulationN(float DeltaTime, int32 Steps);

    /** Set the gravity of the running world (UE axes, cm/s^2). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    bool SetWorldGravity(FVector NewGravity);

    /** Set the substep count of the running world. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics")
    bool SetWorldSubsteps(int32 NewSubsteps);

    // -- Diagnostics --

    /** Most recent panic message caught inside the library on this thread (empty when none). */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Diagnostics")
    FString TakeLastError();

    /** Discard the most recent panic message on this thread. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Diagnostics")
    void ClearLastError();

    /** Library version string. */
    UFUNCTION(BlueprintCallable, Category = "ALICE Physics|Diagnostics")
    static FString GetLibraryVersion();

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
    /** Direction (no unit change): the axis permutation of ToAlice without the cm -> m scale. */
    static AliceVec3 DirToAlice(const FVector& V);
    /** Rotation: the quaternion's vector part permuted like DirToAlice. */
    static AliceQuat QuatToAlice(const FQuat& Q);
    /** Inverse of QuatToAlice. */
    static FQuat QuatFromAlice(const AliceQuat& Q);
    /** Kind + SizeCm (see AddShapedBody) to the C ABI shape. */
    static AlicePhysicsShape ShapeToAlice(int32 Kind, const FVector& SizeCm);
    static int32 IndexOrMinusOne(uint32_t Index);
};
