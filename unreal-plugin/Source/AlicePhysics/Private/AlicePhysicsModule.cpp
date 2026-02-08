// ALICE-Physics UE5 Plugin Module
// Author: Moroya Sakamoto

#include "AlicePhysicsModule.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"

#define LOCTEXT_NAMESPACE "FAlicePhysicsModule"

void FAlicePhysicsModule::StartupModule()
{
    // Load native library
    FString LibPath;

#if PLATFORM_WINDOWS
    LibPath = FPaths::Combine(*FPaths::ProjectPluginsDir(), TEXT("AlicePhysics/ThirdParty/AlicePhysics/lib/Win64/alice_physics.dll"));
#elif PLATFORM_MAC
    LibPath = FPaths::Combine(*FPaths::ProjectPluginsDir(), TEXT("AlicePhysics/ThirdParty/AlicePhysics/lib/macOS/libalice_physics.dylib"));
#elif PLATFORM_LINUX
    LibPath = FPaths::Combine(*FPaths::ProjectPluginsDir(), TEXT("AlicePhysics/ThirdParty/AlicePhysics/lib/Linux/libalice_physics.so"));
#endif

    if (!LibPath.IsEmpty())
    {
        LibraryHandle = FPlatformProcess::GetDllHandle(*LibPath);
        if (LibraryHandle)
        {
            UE_LOG(LogTemp, Log, TEXT("ALICE-Physics: Native library loaded from %s"), *LibPath);
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("ALICE-Physics: Failed to load native library from %s"), *LibPath);
        }
    }
}

void FAlicePhysicsModule::ShutdownModule()
{
    if (LibraryHandle)
    {
        FPlatformProcess::FreeDllHandle(LibraryHandle);
        LibraryHandle = nullptr;
    }
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FAlicePhysicsModule, AlicePhysics)
