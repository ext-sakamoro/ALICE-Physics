# Coverage status

Generated from docs/coverage/*.toml by `python3 scripts/coverage_check.py --write-status`;
do not edit by hand. Each table lists the capabilities a field expects, with what the
crate has today.

| table | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| `docs/coverage/acous.toml` | 101 | 8 | 3 | 3 | 87 | 0 |
| `docs/coverage/am.toml` | 90 | 10 | 0 | 7 | 73 | 0 |
| `docs/coverage/atmos.toml` | 144 | 0 | 1 | 0 | 143 | 0 |
| `docs/coverage/bio.toml` | 135 | 0 | 4 | 0 | 131 | 0 |
| `docs/coverage/cfd.toml` | 117 | 39 | 2 | 6 | 70 | 0 |
| `docs/coverage/chem.toml` | 112 | 7 | 2 | 0 | 103 | 0 |
| `docs/coverage/couple.toml` | 121 | 10 | 3 | 4 | 104 | 0 |
| `docs/coverage/em.toml` | 111 | 20 | 2 | 5 | 84 | 0 |
| `docs/coverage/engine.toml` | 145 | 63 | 9 | 6 | 67 | 0 |
| `docs/coverage/env.toml` | 116 | 20 | 3 | 2 | 91 | 0 |
| `docs/coverage/fem.toml` | 133 | 36 | 0 | 6 | 88 | 3 |
| `docs/coverage/fract.toml` | 106 | 15 | 2 | 4 | 85 | 0 |
| `docs/coverage/mat.toml` | 121 | 23 | 13 | 5 | 80 | 0 |
| `docs/coverage/mbd.toml` | 139 | 36 | 2 | 13 | 88 | 0 |
| `docs/coverage/multiphase.toml` | 86 | 11 | 2 | 3 | 70 | 0 |
| `docs/coverage/nuclear.toml` | 123 | 0 | 1 | 0 | 122 | 0 |
| `docs/coverage/num.toml` | 101 | 22 | 0 | 5 | 73 | 1 |
| `docs/coverage/opt.toml` | 120 | 1 | 0 | 0 | 118 | 1 |
| `docs/coverage/optics.toml` | 119 | 0 | 0 | 0 | 119 | 0 |
| `docs/coverage/orbit.toml` | 109 | 23 | 4 | 3 | 79 | 0 |
| `docs/coverage/part.toml` | 120 | 26 | 2 | 3 | 89 | 0 |
| `docs/coverage/plasma.toml` | 143 | 0 | 4 | 0 | 139 | 0 |
| `docs/coverage/quantum.toml` | 135 | 0 | 0 | 0 | 135 | 0 |
| `docs/coverage/rel.toml` | 112 | 0 | 0 | 0 | 112 | 0 |
| `docs/coverage/rigid.toml` | 119 | 77 | 4 | 8 | 29 | 1 |
| `docs/coverage/sense.toml` | 92 | 19 | 2 | 2 | 69 | 0 |
| `docs/coverage/soft.toml` | 90 | 21 | 4 | 6 | 59 | 0 |
| `docs/coverage/struct.toml` | 134 | 33 | 0 | 10 | 91 | 0 |
| `docs/coverage/therm.toml` | 102 | 17 | 0 | 3 | 82 | 0 |
| **total** | 3396 | 537 | 69 | 104 | 2680 | 6 |

## `docs/coverage/acous.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 5 | 1 | 2 | 0 | 2 | 0 |
| boundary | 11 | 1 | 0 | 0 | 10 | 0 |
| discretisation | 13 | 3 | 0 | 0 | 10 | 0 |
| game-audio | 7 | 1 | 0 | 1 | 5 | 0 |
| geometric | 6 | 0 | 0 | 0 | 6 | 0 |
| governing | 10 | 0 | 0 | 1 | 9 | 0 |
| medium | 8 | 2 | 0 | 1 | 5 | 0 |
| output | 5 | 0 | 0 | 0 | 5 | 0 |
| propagation | 12 | 0 | 0 | 0 | 12 | 0 |
| room | 9 | 0 | 0 | 0 | 9 | 0 |
| source | 8 | 0 | 1 | 0 | 7 | 0 |
| structural | 7 | 0 | 0 | 0 | 7 | 0 |

## `docs/coverage/am.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 7 | 0 | 0 | 0 | 7 | 0 |
| extrusion and bead | 9 | 0 | 0 | 1 | 8 | 0 |
| geometry check | 12 | 3 | 0 | 1 | 8 | 0 |
| layer adhesion | 10 | 3 | 0 | 2 | 5 | 0 |
| orientation | 8 | 1 | 0 | 1 | 6 | 0 |
| pipeline | 3 | 1 | 0 | 0 | 2 | 0 |
| post-processing | 4 | 0 | 0 | 0 | 4 | 0 |
| powder bed fusion | 5 | 0 | 0 | 0 | 5 | 0 |
| slicer interface | 5 | 0 | 0 | 0 | 5 | 0 |
| support | 6 | 2 | 0 | 0 | 4 | 0 |
| thermal history | 8 | 0 | 0 | 0 | 8 | 0 |
| vat photopolymerisation | 4 | 0 | 0 | 0 | 4 | 0 |
| warp and residual stress | 9 | 0 | 0 | 2 | 7 | 0 |

## `docs/coverage/atmos.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| aerosol-activation | 7 | 0 | 0 | 0 | 7 | 0 |
| benchmark | 13 | 0 | 0 | 0 | 13 | 0 |
| convection | 9 | 0 | 1 | 0 | 8 | 0 |
| diagnostics | 3 | 0 | 0 | 0 | 3 | 0 |
| ensemble | 5 | 0 | 0 | 0 | 5 | 0 |
| equations | 10 | 0 | 0 | 0 | 10 | 0 |
| ice-microphysics | 12 | 0 | 0 | 0 | 12 | 0 |
| intervention | 9 | 0 | 0 | 0 | 9 | 0 |
| numerics | 8 | 0 | 0 | 0 | 8 | 0 |
| ocean-coupling | 4 | 0 | 0 | 0 | 4 | 0 |
| radiation | 8 | 0 | 0 | 0 | 8 | 0 |
| rotation-balance | 12 | 0 | 0 | 0 | 12 | 0 |
| surface-boundary-layer | 5 | 0 | 0 | 0 | 5 | 0 |
| thermodynamics | 11 | 0 | 0 | 0 | 11 | 0 |
| tropical-cyclone | 13 | 0 | 0 | 0 | 13 | 0 |
| warm-microphysics | 15 | 0 | 0 | 0 | 15 | 0 |

## `docs/coverage/bio.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 16 | 0 | 0 | 0 | 16 | 0 |
| bioheat-transport | 4 | 0 | 0 | 0 | 4 | 0 |
| blood-flow | 14 | 0 | 1 | 0 | 13 | 0 |
| bone | 9 | 0 | 1 | 0 | 8 | 0 |
| cell-molecular | 10 | 0 | 0 | 0 | 10 | 0 |
| heart | 10 | 0 | 0 | 0 | 10 | 0 |
| impact-injury | 10 | 0 | 0 | 0 | 10 | 0 |
| locomotion | 11 | 0 | 0 | 0 | 11 | 0 |
| muscle | 12 | 0 | 0 | 0 | 12 | 0 |
| musculoskeletal | 9 | 0 | 0 | 0 | 9 | 0 |
| soft-tissue | 16 | 0 | 1 | 0 | 15 | 0 |
| swimming-flying | 9 | 0 | 1 | 0 | 8 | 0 |
| tendon-ligament | 5 | 0 | 0 | 0 | 5 | 0 |

## `docs/coverage/cfd.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| advection | 9 | 3 | 0 | 0 | 6 | 0 |
| benchmark | 21 | 5 | 2 | 0 | 14 | 0 |
| boundary condition | 9 | 5 | 0 | 0 | 4 | 0 |
| discretisation | 10 | 3 | 0 | 0 | 7 | 0 |
| output | 4 | 1 | 0 | 0 | 3 | 0 |
| particle method | 15 | 7 | 0 | 1 | 7 | 0 |
| physics | 10 | 3 | 0 | 0 | 7 | 0 |
| pressure solver | 9 | 4 | 0 | 2 | 3 | 0 |
| pressure-velocity coupling | 6 | 1 | 0 | 0 | 5 | 0 |
| time integration | 8 | 2 | 0 | 1 | 5 | 0 |
| turbulence | 16 | 5 | 0 | 2 | 9 | 0 |

## `docs/coverage/chem.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 1 | 1 | 0 | 9 | 0 |
| combustion | 19 | 2 | 0 | 0 | 17 | 0 |
| electrochemistry | 6 | 0 | 0 | 0 | 6 | 0 |
| equilibrium | 6 | 0 | 0 | 0 | 6 | 0 |
| fire | 8 | 0 | 0 | 0 | 8 | 0 |
| kinetics | 16 | 2 | 1 | 0 | 13 | 0 |
| reactor | 8 | 0 | 0 | 0 | 8 | 0 |
| stoichiometry | 5 | 0 | 0 | 0 | 5 | 0 |
| thermo | 12 | 1 | 0 | 0 | 11 | 0 |
| transport | 9 | 0 | 0 | 0 | 9 | 0 |
| turbulent-combustion | 7 | 0 | 0 | 0 | 7 | 0 |
| visual | 5 | 1 | 0 | 0 | 4 | 0 |

## `docs/coverage/couple.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| aeroelasticity | 14 | 0 | 0 | 1 | 13 | 0 |
| atmosphere-environment | 1 | 0 | 0 | 0 | 1 | 0 |
| benchmark | 15 | 0 | 0 | 0 | 15 | 0 |
| bio-collision | 1 | 0 | 0 | 0 | 1 | 0 |
| bio-multibody | 1 | 0 | 0 | 0 | 1 | 0 |
| bio-nuclear | 1 | 0 | 0 | 0 | 1 | 0 |
| chem-environment | 1 | 0 | 0 | 0 | 1 | 0 |
| cloth-soft-fluid | 7 | 2 | 1 | 0 | 4 | 0 |
| co-simulation | 6 | 1 | 1 | 0 | 4 | 0 |
| electromechanical | 14 | 2 | 0 | 0 | 12 | 0 |
| em-optics | 1 | 0 | 0 | 0 | 1 | 0 |
| framework | 7 | 0 | 1 | 0 | 6 | 0 |
| fsi-formulation | 9 | 1 | 0 | 0 | 8 | 0 |
| fsi-interface | 10 | 1 | 0 | 1 | 8 | 0 |
| multiphase-environment | 1 | 0 | 0 | 0 | 1 | 0 |
| optics-atmosphere | 1 | 0 | 0 | 0 | 1 | 0 |
| particle-atmosphere | 1 | 0 | 0 | 0 | 1 | 0 |
| particle-fluid | 5 | 0 | 0 | 0 | 5 | 0 |
| particle-rigid | 2 | 0 | 0 | 0 | 2 | 0 |
| particle-thermal | 1 | 0 | 0 | 0 | 1 | 0 |
| rigid-fluid | 10 | 2 | 0 | 2 | 6 | 0 |
| shock-structure | 3 | 0 | 0 | 0 | 3 | 0 |
| soft-sensor | 1 | 0 | 0 | 0 | 1 | 0 |
| thermo-atmosphere | 1 | 0 | 0 | 0 | 1 | 0 |
| thermo-mechanical | 6 | 1 | 0 | 0 | 5 | 0 |
| thermo-soft | 1 | 0 | 0 | 0 | 1 | 0 |

## `docs/coverage/em.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 3 | 0 | 0 | 8 | 0 |
| boundary | 12 | 1 | 0 | 1 | 10 | 0 |
| circuit | 6 | 0 | 0 | 0 | 6 | 0 |
| frequency-domain | 7 | 0 | 0 | 0 | 7 | 0 |
| integral-equation | 5 | 0 | 0 | 0 | 5 | 0 |
| material | 14 | 3 | 0 | 1 | 10 | 0 |
| output | 12 | 2 | 0 | 0 | 10 | 0 |
| particle | 5 | 0 | 0 | 0 | 5 | 0 |
| quasi-static | 3 | 0 | 0 | 0 | 3 | 0 |
| source | 9 | 2 | 2 | 0 | 5 | 0 |
| static | 11 | 3 | 0 | 0 | 8 | 0 |
| time-domain | 16 | 6 | 0 | 3 | 7 | 0 |

## `docs/coverage/engine.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| api | 26 | 8 | 5 | 1 | 12 | 0 |
| auxiliary | 34 | 18 | 0 | 2 | 14 | 0 |
| benchmark | 8 | 3 | 2 | 0 | 3 | 0 |
| determinism | 11 | 6 | 2 | 0 | 3 | 0 |
| diagnostics | 15 | 9 | 0 | 0 | 6 | 0 |
| game-engine | 4 | 0 | 0 | 0 | 4 | 0 |
| network | 18 | 7 | 0 | 0 | 11 | 0 |
| state | 18 | 5 | 0 | 2 | 11 | 0 |
| world | 11 | 7 | 0 | 1 | 3 | 0 |

## `docs/coverage/env.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| aerodynamics | 12 | 5 | 0 | 0 | 7 | 0 |
| atmosphere | 13 | 4 | 1 | 0 | 8 | 0 |
| atmospheric-dynamics | 5 | 0 | 0 | 0 | 5 | 0 |
| benchmark | 9 | 3 | 1 | 0 | 5 | 0 |
| boundary-layer | 6 | 0 | 0 | 0 | 6 | 0 |
| environmental-load | 5 | 0 | 0 | 1 | 4 | 0 |
| moist-air | 7 | 0 | 0 | 0 | 7 | 0 |
| ocean-coastal | 5 | 0 | 0 | 0 | 5 | 0 |
| ocean-waves | 17 | 3 | 0 | 0 | 14 | 0 |
| terrain | 9 | 3 | 0 | 0 | 6 | 0 |
| turbulence-gust | 10 | 0 | 0 | 1 | 9 | 0 |
| weather | 9 | 0 | 0 | 0 | 9 | 0 |
| wind-field | 9 | 2 | 1 | 0 | 6 | 0 |

## `docs/coverage/fem.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| adaptivity | 6 | 2 | 0 | 2 | 2 | 0 |
| benchmark | 12 | 4 | 0 | 0 | 8 | 0 |
| constraint | 6 | 1 | 0 | 0 | 5 | 0 |
| contact | 6 | 0 | 0 | 0 | 5 | 1 |
| dynamics | 11 | 3 | 0 | 1 | 7 | 0 |
| eigen | 3 | 0 | 0 | 0 | 3 | 0 |
| element | 17 | 3 | 0 | 0 | 14 | 0 |
| fracture | 5 | 0 | 0 | 0 | 4 | 1 |
| integration | 6 | 1 | 0 | 0 | 4 | 1 |
| kinematics | 5 | 3 | 0 | 0 | 2 | 0 |
| linear_solver | 6 | 2 | 0 | 0 | 4 | 0 |
| load | 7 | 2 | 0 | 0 | 5 | 0 |
| material | 23 | 6 | 0 | 1 | 16 | 0 |
| nonlinear | 8 | 4 | 0 | 1 | 3 | 0 |
| output | 6 | 3 | 0 | 0 | 3 | 0 |
| tangent | 6 | 2 | 0 | 1 | 3 | 0 |

## `docs/coverage/fract.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 10 | 0 | 0 | 0 | 10 | 0 |
| crack-growth | 7 | 0 | 0 | 0 | 7 | 0 |
| csg-destruction | 9 | 6 | 1 | 1 | 1 | 0 |
| cutting | 2 | 0 | 0 | 0 | 2 | 0 |
| damage | 9 | 0 | 0 | 0 | 9 | 0 |
| discrete-crack | 13 | 0 | 0 | 0 | 13 | 0 |
| dynamic-fracture | 4 | 0 | 0 | 0 | 4 | 0 |
| effect-layer-fracture | 11 | 6 | 0 | 2 | 3 | 0 |
| erosion-wear | 14 | 3 | 1 | 1 | 9 | 0 |
| fragmentation | 7 | 0 | 0 | 0 | 7 | 0 |
| lefm | 13 | 0 | 0 | 0 | 13 | 0 |
| phase-field | 7 | 0 | 0 | 0 | 7 | 0 |

## `docs/coverage/mat.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 7 | 1 | 0 | 0 | 6 | 0 |
| contact material | 12 | 3 | 2 | 0 | 7 | 0 |
| damage-fracture data | 3 | 0 | 0 | 0 | 3 | 0 |
| data | 16 | 0 | 6 | 0 | 10 | 0 |
| database | 10 | 2 | 0 | 0 | 8 | 0 |
| elasticity | 12 | 3 | 1 | 1 | 7 | 0 |
| electromagnetic data | 4 | 0 | 1 | 0 | 3 | 0 |
| hyperelasticity | 14 | 6 | 1 | 0 | 7 | 0 |
| plasticity | 16 | 5 | 0 | 2 | 9 | 0 |
| rheology | 12 | 3 | 0 | 2 | 7 | 0 |
| thermal data | 6 | 0 | 2 | 0 | 4 | 0 |
| viscoelasticity | 9 | 0 | 0 | 0 | 9 | 0 |

## `docs/coverage/mbd.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| actuator | 4 | 0 | 0 | 0 | 4 | 0 |
| aero-road | 8 | 2 | 0 | 3 | 3 | 0 |
| benchmark | 14 | 3 | 0 | 1 | 10 | 0 |
| brakes | 7 | 2 | 1 | 0 | 4 | 0 |
| character | 12 | 7 | 0 | 2 | 3 | 0 |
| formulation | 7 | 0 | 0 | 0 | 7 | 0 |
| mechanism | 6 | 1 | 1 | 0 | 4 | 0 |
| powertrain | 12 | 6 | 0 | 0 | 6 | 0 |
| ragdoll | 10 | 5 | 0 | 1 | 4 | 0 |
| robot-control | 5 | 0 | 0 | 1 | 4 | 0 |
| robot-dynamics | 5 | 0 | 0 | 0 | 5 | 0 |
| robot-kinematics | 8 | 0 | 0 | 0 | 8 | 0 |
| rotor | 9 | 2 | 0 | 1 | 6 | 0 |
| steering | 3 | 1 | 0 | 0 | 2 | 0 |
| suspension | 7 | 2 | 0 | 1 | 4 | 0 |
| tyre | 14 | 3 | 0 | 2 | 9 | 0 |
| vehicle | 8 | 2 | 0 | 1 | 5 | 0 |

## `docs/coverage/multiphase.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 12 | 1 | 0 | 0 | 11 | 0 |
| bubble and droplet dynamics | 5 | 0 | 0 | 0 | 5 | 0 |
| combustion flow | 3 | 0 | 0 | 0 | 3 | 0 |
| interface capturing: VOF | 8 | 3 | 0 | 1 | 4 | 0 |
| interface capturing: level set | 12 | 2 | 1 | 1 | 8 | 0 |
| interface capturing: other representations | 4 | 0 | 0 | 0 | 4 | 0 |
| multiphase models (averaged / dispersed) | 6 | 0 | 0 | 0 | 6 | 0 |
| non-Newtonian phases | 2 | 0 | 0 | 0 | 2 | 0 |
| output | 2 | 0 | 0 | 0 | 2 | 0 |
| particle methods: free surface | 7 | 1 | 0 | 0 | 6 | 0 |
| phase change at the interface (flow side) | 4 | 0 | 0 | 0 | 4 | 0 |
| surface tension | 15 | 4 | 1 | 1 | 9 | 0 |
| two-phase momentum | 6 | 0 | 0 | 0 | 6 | 0 |

## `docs/coverage/nuclear.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 0 | 0 | 0 | 11 | 0 |
| charged | 9 | 0 | 0 | 0 | 9 | 0 |
| decay | 13 | 0 | 1 | 0 | 12 | 0 |
| detector | 8 | 0 | 0 | 0 | 8 | 0 |
| dose | 11 | 0 | 0 | 0 | 11 | 0 |
| foundation | 3 | 0 | 0 | 0 | 3 | 0 |
| fusion | 6 | 0 | 0 | 0 | 6 | 0 |
| neutron | 10 | 0 | 0 | 0 | 10 | 0 |
| photon | 10 | 0 | 0 | 0 | 10 | 0 |
| reactor | 17 | 0 | 0 | 0 | 17 | 0 |
| shielding | 7 | 0 | 0 | 0 | 7 | 0 |
| transport | 18 | 0 | 0 | 0 | 18 | 0 |

## `docs/coverage/num.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 9 | 3 | 0 | 0 | 6 | 0 |
| coupling | 7 | 3 | 0 | 0 | 4 | 0 |
| distributed | 6 | 0 | 0 | 0 | 6 | 0 |
| eigen | 9 | 2 | 0 | 0 | 7 | 0 |
| krylov | 10 | 4 | 0 | 1 | 5 | 0 |
| linear_direct | 7 | 1 | 0 | 0 | 6 | 0 |
| matrix_function | 2 | 0 | 0 | 0 | 2 | 0 |
| mesh_adaptivity | 1 | 0 | 0 | 0 | 1 | 0 |
| nonlinear | 9 | 0 | 0 | 0 | 9 | 0 |
| preconditioner | 10 | 2 | 0 | 1 | 7 | 0 |
| representation | 12 | 7 | 0 | 3 | 1 | 1 |
| stationary | 1 | 0 | 0 | 0 | 1 | 0 |
| stochastic | 2 | 0 | 0 | 0 | 2 | 0 |
| time_integration | 14 | 0 | 0 | 0 | 14 | 0 |
| transform | 2 | 0 | 0 | 0 | 2 | 0 |

## `docs/coverage/opt.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| assimilation | 11 | 0 | 0 | 0 | 11 | 0 |
| autodiff | 6 | 0 | 0 | 0 | 6 | 0 |
| benchmark | 12 | 0 | 0 | 0 | 12 | 0 |
| differentiable_sim | 7 | 0 | 0 | 0 | 7 | 0 |
| identification | 9 | 0 | 0 | 0 | 9 | 0 |
| learning | 5 | 0 | 0 | 0 | 4 | 1 |
| problem | 3 | 0 | 0 | 0 | 3 | 0 |
| rom | 11 | 0 | 0 | 0 | 11 | 0 |
| sensitivity | 10 | 0 | 0 | 0 | 10 | 0 |
| shape | 5 | 1 | 0 | 0 | 4 | 0 |
| sizing | 3 | 0 | 0 | 0 | 3 | 0 |
| solver | 12 | 0 | 0 | 0 | 12 | 0 |
| topology | 13 | 0 | 0 | 0 | 13 | 0 |
| uq | 13 | 0 | 0 | 0 | 13 | 0 |

## `docs/coverage/optics.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 12 | 0 | 0 | 0 | 12 | 0 |
| diffraction | 8 | 0 | 0 | 0 | 8 | 0 |
| dispersion | 7 | 0 | 0 | 0 | 7 | 0 |
| elements | 7 | 0 | 0 | 0 | 7 | 0 |
| gaussian-beam | 4 | 0 | 0 | 0 | 4 | 0 |
| geometric | 10 | 0 | 0 | 0 | 10 | 0 |
| imaging | 11 | 0 | 0 | 0 | 11 | 0 |
| interface | 5 | 0 | 0 | 0 | 5 | 0 |
| interference | 6 | 0 | 0 | 0 | 6 | 0 |
| light-matter | 7 | 0 | 0 | 0 | 7 | 0 |
| light-transport | 7 | 0 | 0 | 0 | 7 | 0 |
| polarisation | 5 | 0 | 0 | 0 | 5 | 0 |
| radiative-transfer | 7 | 0 | 0 | 0 | 7 | 0 |
| reflectance | 9 | 0 | 0 | 0 | 9 | 0 |
| scattering | 7 | 0 | 0 | 0 | 7 | 0 |
| spectral | 7 | 0 | 0 | 0 | 7 | 0 |

## `docs/coverage/orbit.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| attitude | 6 | 0 | 0 | 0 | 6 | 0 |
| benchmark | 13 | 3 | 0 | 0 | 10 | 0 |
| elements | 7 | 2 | 0 | 1 | 4 | 0 |
| frames-time | 7 | 0 | 0 | 0 | 7 | 0 |
| maneuver | 9 | 0 | 1 | 0 | 8 | 0 |
| n-body integration | 12 | 3 | 1 | 1 | 7 | 0 |
| n-body law | 7 | 4 | 0 | 1 | 2 | 0 |
| output | 3 | 0 | 1 | 0 | 2 | 0 |
| perturbation | 12 | 1 | 1 | 0 | 10 | 0 |
| propagation | 7 | 1 | 0 | 0 | 6 | 0 |
| restricted three-body | 6 | 0 | 0 | 0 | 6 | 0 |
| surface gravity | 8 | 4 | 0 | 0 | 4 | 0 |
| targeting | 3 | 0 | 0 | 0 | 3 | 0 |
| two-body | 9 | 5 | 0 | 0 | 4 | 0 |

## `docs/coverage/part.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| SPH (beyond single-phase flow) | 5 | 0 | 0 | 0 | 5 | 0 |
| analysis | 4 | 0 | 0 | 0 | 4 | 0 |
| benchmark | 14 | 2 | 1 | 0 | 11 | 0 |
| constraints | 3 | 0 | 0 | 0 | 3 | 0 |
| crowd | 10 | 5 | 0 | 0 | 5 | 0 |
| cutoff | 5 | 3 | 0 | 0 | 2 | 0 |
| discrete element method | 14 | 0 | 0 | 0 | 14 | 0 |
| effect particle system | 7 | 5 | 0 | 0 | 2 | 0 |
| ensemble and thermostat | 11 | 1 | 0 | 1 | 9 | 0 |
| integration | 6 | 3 | 0 | 0 | 3 | 0 |
| long-range | 4 | 0 | 0 | 0 | 4 | 0 |
| many-body and bonded | 8 | 0 | 0 | 0 | 8 | 0 |
| material point method | 7 | 0 | 0 | 0 | 7 | 0 |
| neighbour search and boundary | 10 | 2 | 0 | 2 | 6 | 0 |
| pair potential | 12 | 5 | 1 | 0 | 6 | 0 |

## `docs/coverage/plasma.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 20 | 0 | 2 | 0 | 18 | 0 |
| equilibrium-stability | 9 | 0 | 0 | 0 | 9 | 0 |
| fluid | 9 | 0 | 0 | 0 | 9 | 0 |
| fusion | 6 | 0 | 0 | 0 | 6 | 0 |
| industrial | 5 | 0 | 0 | 0 | 5 | 0 |
| kinetic | 12 | 0 | 0 | 0 | 12 | 0 |
| mhd-numerics | 10 | 0 | 0 | 0 | 10 | 0 |
| parameters | 18 | 0 | 0 | 0 | 18 | 0 |
| particle-in-cell | 17 | 0 | 0 | 0 | 17 | 0 |
| propulsion | 3 | 0 | 0 | 0 | 3 | 0 |
| reconnection | 5 | 0 | 0 | 0 | 5 | 0 |
| single-particle | 9 | 0 | 1 | 0 | 8 | 0 |
| space | 5 | 0 | 0 | 0 | 5 | 0 |
| strongly-coupled | 2 | 0 | 1 | 0 | 1 | 0 |
| waves | 13 | 0 | 0 | 0 | 13 | 0 |

## `docs/coverage/quantum.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| approximation | 8 | 0 | 0 | 0 | 8 | 0 |
| benchmark | 14 | 0 | 0 | 0 | 14 | 0 |
| closed-form | 16 | 0 | 0 | 0 | 16 | 0 |
| foundation | 8 | 0 | 0 | 0 | 8 | 0 |
| many-body | 14 | 0 | 0 | 0 | 14 | 0 |
| open-system | 10 | 0 | 0 | 0 | 10 | 0 |
| quantum-information | 12 | 0 | 0 | 0 | 12 | 0 |
| scattering | 8 | 0 | 0 | 0 | 8 | 0 |
| semiclassical | 7 | 0 | 0 | 0 | 7 | 0 |
| spin | 11 | 0 | 0 | 0 | 11 | 0 |
| stationary | 9 | 0 | 0 | 0 | 9 | 0 |
| statistics | 8 | 0 | 0 | 0 | 8 | 0 |
| time-dependent | 10 | 0 | 0 | 0 | 10 | 0 |

## `docs/coverage/rel.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 15 | 0 | 0 | 0 | 15 | 0 |
| charged-particle | 8 | 0 | 0 | 0 | 8 | 0 |
| covariant-em | 5 | 0 | 0 | 0 | 5 | 0 |
| gr-closed-form | 11 | 0 | 0 | 0 | 11 | 0 |
| metric-geodesic | 11 | 0 | 0 | 0 | 11 | 0 |
| numerical-relativity | 8 | 0 | 0 | 0 | 8 | 0 |
| post-newtonian | 9 | 0 | 0 | 0 | 9 | 0 |
| rel-fluid | 9 | 0 | 0 | 0 | 9 | 0 |
| rel-quantum | 3 | 0 | 0 | 0 | 3 | 0 |
| sr-dynamics | 13 | 0 | 0 | 0 | 13 | 0 |
| sr-kinematics | 15 | 0 | 0 | 0 | 15 | 0 |
| time-scales | 2 | 0 | 0 | 0 | 2 | 0 |
| units | 3 | 0 | 0 | 0 | 3 | 0 |

## `docs/coverage/rigid.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| 2d | 2 | 2 | 0 | 0 | 0 | 0 |
| benchmark | 14 | 5 | 0 | 0 | 8 | 1 |
| broadphase | 5 | 3 | 0 | 0 | 2 | 0 |
| ccd | 6 | 4 | 0 | 0 | 2 | 0 |
| contact | 17 | 10 | 1 | 2 | 4 | 0 |
| dynamics | 15 | 12 | 0 | 0 | 3 | 0 |
| joint | 19 | 17 | 1 | 0 | 1 | 0 |
| narrowphase | 18 | 11 | 2 | 3 | 2 | 0 |
| query | 7 | 3 | 0 | 2 | 2 | 0 |
| sleep | 3 | 3 | 0 | 0 | 0 | 0 |
| solver | 13 | 7 | 0 | 1 | 5 | 0 |

## `docs/coverage/sense.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 4 | 1 | 0 | 6 | 0 |
| contact-force | 6 | 1 | 0 | 2 | 3 | 0 |
| control | 14 | 2 | 0 | 0 | 12 | 0 |
| estimation | 17 | 1 | 0 | 0 | 16 | 0 |
| inertial | 8 | 4 | 0 | 0 | 4 | 0 |
| noise | 12 | 2 | 0 | 0 | 10 | 0 |
| proprio-nav | 7 | 0 | 1 | 0 | 6 | 0 |
| range | 14 | 5 | 0 | 0 | 9 | 0 |
| sysid | 3 | 0 | 0 | 0 | 3 | 0 |

## `docs/coverage/soft.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| attachment / external force | 6 | 1 | 0 | 2 | 3 | 0 |
| benchmark | 8 | 3 | 1 | 0 | 4 | 0 |
| cloth model | 14 | 2 | 0 | 0 | 12 | 0 |
| collision | 16 | 4 | 0 | 2 | 10 | 0 |
| integration / output | 4 | 2 | 0 | 0 | 2 | 0 |
| rod / rope / hair | 12 | 3 | 0 | 1 | 8 | 0 |
| soft body | 9 | 0 | 3 | 1 | 5 | 0 |
| solver | 12 | 3 | 0 | 0 | 9 | 0 |
| time integration / damping | 4 | 2 | 0 | 0 | 2 | 0 |
| topology change | 5 | 1 | 0 | 0 | 4 | 0 |

## `docs/coverage/struct.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| beam | 14 | 2 | 0 | 0 | 12 | 0 |
| buckling | 11 | 3 | 0 | 1 | 7 | 0 |
| contact-stress | 3 | 1 | 0 | 0 | 2 | 0 |
| creep | 9 | 1 | 0 | 2 | 6 | 0 |
| fatigue | 17 | 3 | 0 | 2 | 12 | 0 |
| joint | 13 | 4 | 0 | 1 | 8 | 0 |
| laminate | 18 | 8 | 0 | 0 | 10 | 0 |
| plate-shell | 8 | 0 | 0 | 0 | 8 | 0 |
| section | 8 | 2 | 0 | 0 | 6 | 0 |
| stress-concentration | 8 | 2 | 0 | 2 | 4 | 0 |
| synthesis | 3 | 0 | 0 | 1 | 2 | 0 |
| thermal-stress | 4 | 1 | 0 | 1 | 2 | 0 |
| torsion | 4 | 0 | 0 | 0 | 4 | 0 |
| vibration | 14 | 6 | 0 | 0 | 8 | 0 |

## `docs/coverage/therm.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 2 | 0 | 0 | 9 | 0 |
| boundary condition | 10 | 1 | 0 | 0 | 9 | 0 |
| closed-form conduction | 10 | 1 | 0 | 0 | 9 | 0 |
| conduction | 18 | 8 | 0 | 1 | 9 | 0 |
| convection (closed form) | 12 | 0 | 0 | 0 | 12 | 0 |
| coupling (thermal side) | 6 | 3 | 0 | 0 | 3 | 0 |
| fin / heat exchanger | 5 | 0 | 0 | 0 | 5 | 0 |
| heat source | 8 | 1 | 0 | 1 | 6 | 0 |
| output | 3 | 1 | 0 | 0 | 2 | 0 |
| phase change | 9 | 0 | 0 | 1 | 8 | 0 |
| radiation | 10 | 0 | 0 | 0 | 10 | 0 |
