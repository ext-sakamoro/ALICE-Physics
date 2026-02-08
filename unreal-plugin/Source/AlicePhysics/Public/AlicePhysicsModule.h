// ALICE-Physics UE5 Plugin Module Header
// Author: Moroya Sakamoto

#pragma once

#include "Modules/ModuleManager.h"

class FAlicePhysicsModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

private:
    void* LibraryHandle = nullptr;
};
