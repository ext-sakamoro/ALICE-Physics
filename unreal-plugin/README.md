# ALICE-Physics UE5 Plugin

Deterministic 128-bit fixed-point physics engine for Unreal Engine 5.

## Setup

1. Copy the `AlicePhysics` folder to your project's `Plugins/` directory
2. The native library (.dll/.dylib/.so) should be in `ThirdParty/AlicePhysics/lib/<Platform>/`
3. Enable the plugin in Edit > Plugins

## Usage

Add `UAlicePhysicsWorldComponent` to any Actor:

```cpp
// Blueprint or C++
UAlicePhysicsWorldComponent* Physics = GetOwner()->FindComponentByClass<UAlicePhysicsWorldComponent>();

int32 BodyId = Physics->AddDynamicBody(FVector(0, 0, 1000), 1.0f);
Physics->ApplyImpulse(BodyId, FVector(0, 0, 500));

// State serialization for rollback netcode
TArray<uint8> State = Physics->SerializeState();
Physics->DeserializeState(State);
```

## Coordinate System

- UE5: X-forward, Y-right, Z-up (centimeters)
- ALICE: X-right, Y-up, Z-forward (meters)

Conversion is handled automatically by the component.

## Author

Moroya Sakamoto
