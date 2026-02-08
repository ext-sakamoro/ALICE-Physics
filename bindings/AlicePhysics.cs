// ALICE-Physics: Deterministic 128-bit Fixed-Point Physics Engine
// Unity C# Bindings (P/Invoke)
//
// Author: Moroya Sakamoto
// License: AGPL-3.0
//
// Usage:
//   var world = new AlicePhysicsWorld();
//   uint bodyId = world.AddDynamicBody(new Vector3(0, 10, 0), 1.0);
//   world.Step(1.0 / 60.0);
//   Vector3 pos = world.GetBodyPosition(bodyId);
//   world.Dispose();

using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace AlicePhysics
{
    // ========================================================================
    // Native Types (must match alice_physics.h)
    // ========================================================================

    [StructLayout(LayoutKind.Sequential)]
    public struct AliceVec3
    {
        public double x;
        public double y;
        public double z;

        public AliceVec3(double x, double y, double z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static AliceVec3 FromVector3(Vector3 v)
        {
            return new AliceVec3(v.x, v.y, v.z);
        }

        public Vector3 ToVector3()
        {
            return new Vector3((float)x, (float)y, (float)z);
        }

        public static readonly AliceVec3 Zero = new AliceVec3(0, 0, 0);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct AliceQuat
    {
        public double x;
        public double y;
        public double z;
        public double w;

        public AliceQuat(double x, double y, double z, double w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        public Quaternion ToQuaternion()
        {
            return new Quaternion((float)x, (float)y, (float)z, (float)w);
        }

        public static readonly AliceQuat Identity = new AliceQuat(0, 0, 0, 1);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct AlicePhysicsConfig
    {
        public uint substeps;
        public uint iterations;
        public double gravityX;
        public double gravityY;
        public double gravityZ;
        public double damping;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct AliceBodyInfo
    {
        public AliceVec3 position;
        public AliceVec3 velocity;
        public AliceVec3 angularVelocity;
        public AliceQuat rotation;
        public double invMass;
        public byte isStatic;
        public byte isSensor;
    }

    // ========================================================================
    // P/Invoke Declarations
    // ========================================================================

    internal static class Native
    {
#if UNITY_IOS && !UNITY_EDITOR
        private const string DLL = "__Internal";
#else
        private const string DLL = "alice_physics";
#endif

        // World lifecycle
        [DllImport(DLL)] public static extern IntPtr alice_physics_world_create();
        [DllImport(DLL)] public static extern IntPtr alice_physics_world_create_with_config(AlicePhysicsConfig config);
        [DllImport(DLL)] public static extern void alice_physics_world_destroy(IntPtr world);
        [DllImport(DLL)] public static extern void alice_physics_world_step(IntPtr world, double dt);
        [DllImport(DLL)] public static extern uint alice_physics_world_body_count(IntPtr world);

        // Body management
        [DllImport(DLL)] public static extern uint alice_physics_body_add_dynamic(IntPtr world, AliceVec3 position, double mass);
        [DllImport(DLL)] public static extern uint alice_physics_body_add_static(IntPtr world, AliceVec3 position);
        [DllImport(DLL)] public static extern uint alice_physics_body_add_sensor(IntPtr world, AliceVec3 position);
        [DllImport(DLL)] public static extern byte alice_physics_body_get_info(IntPtr world, uint bodyId, out AliceBodyInfo info);
        [DllImport(DLL)] public static extern byte alice_physics_body_get_position(IntPtr world, uint bodyId, out AliceVec3 position);
        [DllImport(DLL)] public static extern byte alice_physics_body_set_position(IntPtr world, uint bodyId, AliceVec3 position);
        [DllImport(DLL)] public static extern byte alice_physics_body_get_velocity(IntPtr world, uint bodyId, out AliceVec3 velocity);
        [DllImport(DLL)] public static extern byte alice_physics_body_set_velocity(IntPtr world, uint bodyId, AliceVec3 velocity);
        [DllImport(DLL)] public static extern byte alice_physics_body_get_rotation(IntPtr world, uint bodyId, out AliceQuat rotation);
        [DllImport(DLL)] public static extern byte alice_physics_body_set_restitution(IntPtr world, uint bodyId, double restitution);
        [DllImport(DLL)] public static extern byte alice_physics_body_set_friction(IntPtr world, uint bodyId, double friction);
        [DllImport(DLL)] public static extern byte alice_physics_body_apply_impulse(IntPtr world, uint bodyId, AliceVec3 impulse);
        [DllImport(DLL)] public static extern byte alice_physics_body_apply_impulse_at(IntPtr world, uint bodyId, AliceVec3 impulse, AliceVec3 point);

        // Config
        [DllImport(DLL)] public static extern AlicePhysicsConfig alice_physics_config_default();
        [DllImport(DLL)] public static extern void alice_physics_world_set_gravity(IntPtr world, double x, double y, double z);
        [DllImport(DLL)] public static extern void alice_physics_world_set_substeps(IntPtr world, uint substeps);

        // State serialization
        [DllImport(DLL)] public static extern IntPtr alice_physics_state_serialize(IntPtr world, out uint len);
        [DllImport(DLL)] public static extern byte alice_physics_state_deserialize(IntPtr world, IntPtr data, uint len);
        [DllImport(DLL)] public static extern void alice_physics_state_free(IntPtr data, uint len);

        // Version
        [DllImport(DLL)] public static extern IntPtr alice_physics_version();
    }

    // ========================================================================
    // High-Level API
    // ========================================================================

    /// <summary>
    /// Deterministic physics world. Uses 128-bit fixed-point internally.
    /// Implements IDisposable for safe native memory management.
    /// </summary>
    public class AlicePhysicsWorld : IDisposable
    {
        private IntPtr _ptr;
        private bool _disposed;

        /// <summary>Create a world with default config (gravity=-10, substeps=8).</summary>
        public AlicePhysicsWorld()
        {
            _ptr = Native.alice_physics_world_create();
        }

        /// <summary>Create a world with custom config.</summary>
        public AlicePhysicsWorld(AlicePhysicsConfig config)
        {
            _ptr = Native.alice_physics_world_create_with_config(config);
        }

        ~AlicePhysicsWorld()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && _ptr != IntPtr.Zero)
            {
                Native.alice_physics_world_destroy(_ptr);
                _ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(AlicePhysicsWorld));
        }

        /// <summary>Native pointer (for advanced use).</summary>
        public IntPtr NativePtr => _ptr;

        // -- Simulation --

        /// <summary>Step simulation by dt seconds.</summary>
        public void Step(double dt)
        {
            ThrowIfDisposed();
            Native.alice_physics_world_step(_ptr, dt);
        }

        /// <summary>Number of bodies.</summary>
        public uint BodyCount
        {
            get { ThrowIfDisposed(); return Native.alice_physics_world_body_count(_ptr); }
        }

        // -- Body creation --

        /// <summary>Add a dynamic body.</summary>
        public uint AddDynamicBody(Vector3 position, double mass)
        {
            ThrowIfDisposed();
            return Native.alice_physics_body_add_dynamic(_ptr, AliceVec3.FromVector3(position), mass);
        }

        /// <summary>Add a static (immovable) body.</summary>
        public uint AddStaticBody(Vector3 position)
        {
            ThrowIfDisposed();
            return Native.alice_physics_body_add_static(_ptr, AliceVec3.FromVector3(position));
        }

        /// <summary>Add a sensor (trigger) body.</summary>
        public uint AddSensorBody(Vector3 position)
        {
            ThrowIfDisposed();
            return Native.alice_physics_body_add_sensor(_ptr, AliceVec3.FromVector3(position));
        }

        // -- Body state --

        /// <summary>Get full body info snapshot.</summary>
        public AliceBodyInfo GetBodyInfo(uint bodyId)
        {
            ThrowIfDisposed();
            AliceBodyInfo info;
            Native.alice_physics_body_get_info(_ptr, bodyId, out info);
            return info;
        }

        /// <summary>Get body position as Vector3.</summary>
        public Vector3 GetBodyPosition(uint bodyId)
        {
            ThrowIfDisposed();
            AliceVec3 pos;
            Native.alice_physics_body_get_position(_ptr, bodyId, out pos);
            return pos.ToVector3();
        }

        /// <summary>Set body position.</summary>
        public void SetBodyPosition(uint bodyId, Vector3 position)
        {
            ThrowIfDisposed();
            Native.alice_physics_body_set_position(_ptr, bodyId, AliceVec3.FromVector3(position));
        }

        /// <summary>Get body velocity as Vector3.</summary>
        public Vector3 GetBodyVelocity(uint bodyId)
        {
            ThrowIfDisposed();
            AliceVec3 vel;
            Native.alice_physics_body_get_velocity(_ptr, bodyId, out vel);
            return vel.ToVector3();
        }

        /// <summary>Set body velocity.</summary>
        public void SetBodyVelocity(uint bodyId, Vector3 velocity)
        {
            ThrowIfDisposed();
            Native.alice_physics_body_set_velocity(_ptr, bodyId, AliceVec3.FromVector3(velocity));
        }

        /// <summary>Get body rotation as Quaternion.</summary>
        public Quaternion GetBodyRotation(uint bodyId)
        {
            ThrowIfDisposed();
            AliceQuat rot;
            Native.alice_physics_body_get_rotation(_ptr, bodyId, out rot);
            return rot.ToQuaternion();
        }

        /// <summary>Set body restitution (bounciness).</summary>
        public void SetBodyRestitution(uint bodyId, double restitution)
        {
            ThrowIfDisposed();
            Native.alice_physics_body_set_restitution(_ptr, bodyId, restitution);
        }

        /// <summary>Set body friction.</summary>
        public void SetBodyFriction(uint bodyId, double friction)
        {
            ThrowIfDisposed();
            Native.alice_physics_body_set_friction(_ptr, bodyId, friction);
        }

        /// <summary>Apply impulse at center of mass.</summary>
        public void ApplyImpulse(uint bodyId, Vector3 impulse)
        {
            ThrowIfDisposed();
            Native.alice_physics_body_apply_impulse(_ptr, bodyId, AliceVec3.FromVector3(impulse));
        }

        /// <summary>Apply impulse at a world-space point.</summary>
        public void ApplyImpulseAt(uint bodyId, Vector3 impulse, Vector3 point)
        {
            ThrowIfDisposed();
            Native.alice_physics_body_apply_impulse_at(_ptr, bodyId, AliceVec3.FromVector3(impulse), AliceVec3.FromVector3(point));
        }

        // -- Config --

        /// <summary>Set gravity.</summary>
        public void SetGravity(Vector3 gravity)
        {
            ThrowIfDisposed();
            Native.alice_physics_world_set_gravity(_ptr, gravity.x, gravity.y, gravity.z);
        }

        /// <summary>Set substeps per frame.</summary>
        public void SetSubsteps(uint substeps)
        {
            ThrowIfDisposed();
            Native.alice_physics_world_set_substeps(_ptr, substeps);
        }

        // -- State serialization (rollback netcode) --

        /// <summary>Serialize world state to byte array.</summary>
        public byte[] SerializeState()
        {
            ThrowIfDisposed();
            uint len;
            IntPtr data = Native.alice_physics_state_serialize(_ptr, out len);
            if (data == IntPtr.Zero || len == 0) return Array.Empty<byte>();

            byte[] result = new byte[len];
            Marshal.Copy(data, result, 0, (int)len);
            Native.alice_physics_state_free(data, len);
            return result;
        }

        /// <summary>Deserialize (restore) world state from byte array.</summary>
        public bool DeserializeState(byte[] data)
        {
            ThrowIfDisposed();
            if (data == null || data.Length == 0) return false;

            IntPtr buf = Marshal.AllocHGlobal(data.Length);
            try
            {
                Marshal.Copy(data, 0, buf, data.Length);
                return Native.alice_physics_state_deserialize(_ptr, buf, (uint)data.Length) != 0;
            }
            finally
            {
                Marshal.FreeHGlobal(buf);
            }
        }

        // -- Version --

        /// <summary>Get native library version string.</summary>
        public static string Version
        {
            get
            {
                IntPtr ptr = Native.alice_physics_version();
                return Marshal.PtrToStringAnsi(ptr) ?? "unknown";
            }
        }
    }
}
