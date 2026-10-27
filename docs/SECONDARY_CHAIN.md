# Secondary motion chains

`src/secondary_chain.rs` moves the parts of a character that the animation does not drive
directly: hair, tails, ribbons, skirt panels. This page defines the model in physical terms, so the
module can be read, checked and tuned without reference to any other system.

## Model

A chain is a tree of point masses. Each particle is the tip of a rigid bone of fixed length hanging
from its parent particle, or, for a root, from an anchor point that the animation moves. All masses
are equal and only relative quantities enter, so no mass appears in the settings.

Every step of length `dt` reads, for each particle, its **animated position** (where the pose alone
would put it) and the **animated frame** of its bone (the world rotation the pose gives the bone),
and the anchor's translation since the last step. Then it updates the particles in
order, parents first. Each particle is updated once per step, using its parent's already updated
position (a Gauss-Seidel sweep down the tree).

The state of a particle is its position now, `p`, and one step ago, `q`. Velocity is implicit,
`(p − q) / dt`, as in position Verlet integration.

## One step of one particle

With `parent` the parent's simulated position (the anchor for a root) and `link` the animated
offset of the particle from its animated parent:

1. **Inertia.** `x = p + momentum · (p − q) + immobile · anchor_delta`
2. **Gravity.** `x += g · gravity · (1 − gravity_falloff · max(0, r̂ · ĝ)) · dt²`, where `r̂` is the
   rest direction (below) and `ĝ` the direction of gravity.
3. **Pull to the pose.** Target `t = parent + link`, then `x += (t − x) · pull`.
4. **Stiffness.** Target `t' = parent + R · link`, then `x += (t' − x) · stiffness`. Here `R` is the
   rotation that takes the parent bone from its animated direction to its simulated direction, and
   `R · link` is the rest direction `r` (for a root, `R` is the identity).
5. **Constraints.** `q = p`, `p = x`. Then, `iterations` times and in this order: the bone length,
   the angle limit and the colliders. Last comes the bone length once more.

## The settings

All settings except `radius` and the limit angles lie in `[0, 1]` and act **per step**.
`SecondaryChain::new` refuses anything outside its range.

| setting | physical meaning |
|---|---|
| `momentum` | The share of last step's displacement that is kept. `1 − momentum` is the velocity lost per step to drag. Damping rate `γ = −ln(momentum) / dt` (`≈ (1 − momentum) / dt` near 1). |
| `pull` | A spring toward the pose, as the share of the remaining distance closed per step. The target translates with the simulated parent but keeps the animated orientation. |
| `stiffness` | The same spring, but the target turns with the parent bone. It keeps the local shape of the chain when the parent swings away from its pose. |
| `gravity` | The share of the gravity vector applied. |
| `gravity_falloff` | Reduces gravity by up to this share when the rest direction already points along gravity. A bone that hangs in its pose then sags less than one held out sideways. |
| `immobile` | The share of the anchor's translation the particle takes rigidly within the same step. At 1 a translating root carries the particle without lag; rotation still lags. |
| `limit` | The directions the bone may take from its parent, around its rest direction: none, a cone (`Angle`), a plane with a swing range (`Hinge`), or an ellipse of angles (`Polar`). |
| `radius` | The particle's radius against the capsule colliders. |
| `rest_length` | The bone's length, held exactly after every step. |

Settings that vary along a chain (stronger pull near the root, more sway at the tip) are given per
particle. `curve_value` evaluates a curve of keys (cubic Hermite between keys, held outside them),
and the caller scales a setting by it at the particle's place along the chain.

## Dynamics in continuous terms

Take a particle with no constraint active, no gravity and a fixed target `T` (the pull target).
Write `e = p − T`. Steps 1 and 3 give the exact recurrence

```
e[n+1] = (1 − pull) · ((1 + momentum) · e[n] − momentum · e[n−1])
```

Its characteristic roots `λ` solve

```
λ² − (1 − pull)(1 + momentum) λ + (1 − pull) momentum = 0
```

- **Decay.** The offset decays if and only if `pull > 0`. With `pull = 0` one root is exactly 1:
  nothing restores the particle, and an offset stays.
- **Damped spring.** Each root corresponds to a continuous-time rate `s = ln(λ) / dt`. When the roots
  are complex, the motion is a damped oscillation `e(t) ∝ e^(Re s · t) cos(Im s · t)` with
  - decay rate `Re s = ln((1 − pull) · momentum) / (2 dt)` (exact, since `|λ|² = (1 − pull) · momentum`)
  - angular frequency `Im s ≈ √pull / dt` for small `pull` (`pull = 0.01`, `momentum = 0.95`,
    `dt = 1/60`: exact `5.810 rad/s`, approximation `6.0`)

  Drag therefore comes from both settings: `momentum` alone does not set the damping.
- **Stiffness alongside pull.** When `stiffness` is also set, step 4 is a second relaxation of the
  same form. With the two targets equal, the combined share is `1 − (1 − pull)(1 − stiffness)`.

Under gravity alone (no pull), each step adds `g · gravity · share · dt²` to the displacement. The
per-step displacement grows geometrically to `g · gravity · share · dt² / (1 − momentum)`, which is
a terminal velocity of `g · gravity · share · dt / (1 − momentum)`.

**The settings are per step, so the motion depends on `dt`.** To get the same continuous motion at
another step length `dt' = r · dt`, raise the roots to the power `r` and solve back:

```
S = λ₁^r + λ₂^r,  P = ((1 − pull) · momentum)^r
pull' = 1 − (S − P),  momentum' = P / (S − P)
```

This holds whenever the roots are complex or positive real. Both continuous-time rates are then
preserved exactly; checked numerically for `r = 0.5` and `2` over oscillating and overdamped
settings, with the converted settings staying in `[0, 1]`. Use one fixed `dt` where runs must agree
bit for bit.

## Constraints

- **Bone length.** The particle is projected onto the sphere of radius `rest_length` around its
  parent. A particle already at that distance is left as it is.
- **Angle limit.** The bone direction is projected onto the allowed set around the rest direction.
  The hinge axis and the polar reference axis are given in the bone's own frame. Each step turns
  them to the world by the bone's animated frame, then by the parent link's turn from its pose. A
  limit therefore follows the character when it turns, and follows a twist of the bone about its
  own axis, which positions alone cannot show.
  - the cone boundary for `Angle`
  - the hinge plane, then the swing range, for `Hinge`
  - the ellipse boundary in the two angles, for `Polar`

  The length is kept.
- **Colliders.** A particle closer than `radius + capsule radius` to a capsule's segment is pushed out
  along the line from the closest point on the segment. A capsule that holds the link's animated parent end
  (closer to its segment than the capsule's own radius) does not act on that link: the bone is
  attached inside that body, and pushing its tip out would only lift it off its pose. The test
  uses the animated parent end, so it does not change while the simulation moves the parent.
- **Order and passes.** Length, limit and colliders run `iterations` times, then the length runs once
  more, so the bone length holds exactly after every step. That last pass can pull a pushed-out
  particle slightly back into a collider: the depth goes as `push² / (2 · rest_length)` and shrinks
  with more passes. In a test where the bone swings onto a capsule, the measured depth is about
  `2e-5` with 1 pass, `4e-7` with 2 and `2e-10` with 4.

## Invariants

- A chain standing at its pose with no gravity stays there bit for bit: every stage leaves a particle
  that already satisfies it untouched.
- Every bone has exactly its rest length after every step.
- The arithmetic is `Fix128` (the trigonometry is CORDIC), so a run gives the same bits on every
  platform.

## Oracles

`tests/secondary_chain_oracle.rs` checks each stage against values solved independently in `f64`.
It covers:

- the fixed point of pull against gravity on the bone's circle, with and without falloff
- pull and stiffness aiming from the simulated parent
- inertia, `immobile`, every angle limit and the rest invariant
- the hinge and polar axes under a turned character, under a twist about the bone, below a moved
  parent, and with both turns at once (which also fixes their order)
- the collider depth bound above
- a capsule holding the link's attachment leaves it bit for bit as a chain with no capsule; one that
  does not hold it still acts; the attachment is judged at its animated place; the capsule's own
  radius decides (not plus the particle's)
- the refusal of out-of-range settings

Each stage was broken on purpose (sign, missing factor, wrong target, skipped pass), and each
broken version fails at least one of these tests.
