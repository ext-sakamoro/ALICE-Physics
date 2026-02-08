// ALICE-Physics UE5 Plugin Build Configuration
// Author: Moroya Sakamoto

using UnrealBuildTool;
using System.IO;

public class AlicePhysics : ModuleRules
{
    public AlicePhysics(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
        });

        // ThirdParty library path
        string ThirdPartyPath = Path.Combine(ModuleDirectory, "..", "..", "ThirdParty", "AlicePhysics");
        string IncludePath = Path.Combine(ThirdPartyPath, "include");
        string LibPath = Path.Combine(ThirdPartyPath, "lib");

        PublicIncludePaths.Add(IncludePath);

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string DllPath = Path.Combine(LibPath, "Win64", "alice_physics.dll");
            PublicAdditionalLibraries.Add(DllPath);
            RuntimeDependencies.Add(DllPath);
            PublicDelayLoadDLLs.Add("alice_physics.dll");
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            string DylibPath = Path.Combine(LibPath, "macOS", "libalice_physics.dylib");
            PublicAdditionalLibraries.Add(DylibPath);
            RuntimeDependencies.Add(DylibPath);
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            string SoPath = Path.Combine(LibPath, "Linux", "libalice_physics.so");
            PublicAdditionalLibraries.Add(SoPath);
            RuntimeDependencies.Add(SoPath);
        }
    }
}
