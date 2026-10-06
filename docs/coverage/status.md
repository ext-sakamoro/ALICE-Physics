# Coverage status

Generated from docs/coverage/*.toml by `python3 scripts/coverage_check.py --write-status`;
do not edit by hand. Each table lists the capabilities a field expects, with what the
crate has today.

| table | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| `docs/coverage/acous.toml` | 103 | 8 | 3 | 3 | 89 | 0 |
| `docs/coverage/am.toml` | 93 | 10 | 0 | 7 | 76 | 0 |
| `docs/coverage/atmos.toml` | 144 | 0 | 1 | 0 | 143 | 0 |
| `docs/coverage/bio.toml` | 141 | 0 | 4 | 0 | 137 | 0 |
| `docs/coverage/cfd.toml` | 121 | 39 | 2 | 6 | 74 | 0 |
| `docs/coverage/chem.toml` | 140 | 7 | 2 | 0 | 131 | 0 |
| `docs/coverage/couple.toml` | 123 | 10 | 3 | 4 | 106 | 0 |
| `docs/coverage/em.toml` | 113 | 20 | 2 | 5 | 86 | 0 |
| `docs/coverage/engine.toml` | 153 | 64 | 10 | 7 | 72 | 0 |
| `docs/coverage/env.toml` | 152 | 20 | 3 | 2 | 127 | 0 |
| `docs/coverage/fem.toml` | 137 | 36 | 0 | 6 | 92 | 3 |
| `docs/coverage/fract.toml` | 112 | 15 | 2 | 4 | 91 | 0 |
| `docs/coverage/geo.toml` | 125 | 0 | 2 | 0 | 123 | 0 |
| `docs/coverage/geotech.toml` | 89 | 0 | 2 | 0 | 87 | 0 |
| `docs/coverage/mat.toml` | 146 | 23 | 13 | 5 | 105 | 0 |
| `docs/coverage/mbd.toml` | 161 | 36 | 2 | 13 | 110 | 0 |
| `docs/coverage/mfg.toml` | 45 | 0 | 0 | 0 | 45 | 0 |
| `docs/coverage/multiphase.toml` | 92 | 11 | 2 | 3 | 76 | 0 |
| `docs/coverage/nonlin.toml` | 96 | 1 | 2 | 0 | 93 | 0 |
| `docs/coverage/nuclear.toml` | 133 | 0 | 1 | 0 | 132 | 0 |
| `docs/coverage/num.toml` | 124 | 26 | 0 | 5 | 92 | 1 |
| `docs/coverage/opt.toml` | 120 | 1 | 0 | 0 | 118 | 1 |
| `docs/coverage/optics.toml` | 121 | 0 | 0 | 0 | 121 | 0 |
| `docs/coverage/orbit.toml` | 138 | 23 | 4 | 3 | 108 | 0 |
| `docs/coverage/part.toml` | 122 | 26 | 2 | 3 | 91 | 0 |
| `docs/coverage/plasma.toml` | 156 | 0 | 4 | 0 | 152 | 0 |
| `docs/coverage/quantum.toml` | 169 | 0 | 0 | 0 | 169 | 0 |
| `docs/coverage/rel.toml` | 112 | 0 | 0 | 0 | 112 | 0 |
| `docs/coverage/rigid.toml` | 121 | 78 | 4 | 7 | 31 | 1 |
| `docs/coverage/sense.toml` | 92 | 19 | 2 | 2 | 69 | 0 |
| `docs/coverage/shock.toml` | 115 | 29 | 7 | 2 | 77 | 0 |
| `docs/coverage/soft.toml` | 96 | 21 | 4 | 6 | 65 | 0 |
| `docs/coverage/stat.toml` | 107 | 0 | 0 | 0 | 107 | 0 |
| `docs/coverage/struct.toml` | 152 | 33 | 0 | 10 | 109 | 0 |
| `docs/coverage/therm.toml` | 128 | 17 | 0 | 3 | 108 | 0 |
| **total** | 4292 | 573 | 83 | 106 | 3524 | 6 |

## `docs/coverage/acous.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 5 | 1 | 2 | 0 | 2 | 0 |
| boundary | 11 | 1 | 0 | 0 | 10 | 0 |
| discretisation | 13 | 3 | 0 | 0 | 10 | 0 |
| game-audio | 9 | 1 | 0 | 1 | 7 | 0 |
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
| slicer interface | 8 | 0 | 0 | 0 | 8 | 0 |
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
| bioheat-transport | 6 | 0 | 0 | 0 | 6 | 0 |
| blood-flow | 14 | 0 | 1 | 0 | 13 | 0 |
| bone | 9 | 0 | 1 | 0 | 8 | 0 |
| cell-molecular | 10 | 0 | 0 | 0 | 10 | 0 |
| heart | 12 | 0 | 0 | 0 | 12 | 0 |
| impact-injury | 12 | 0 | 0 | 0 | 12 | 0 |
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
| physics | 14 | 3 | 0 | 0 | 11 | 0 |
| pressure solver | 9 | 4 | 0 | 2 | 3 | 0 |
| pressure-velocity coupling | 6 | 1 | 0 | 0 | 5 | 0 |
| time integration | 8 | 2 | 0 | 1 | 5 | 0 |
| turbulence | 16 | 5 | 0 | 2 | 9 | 0 |

## `docs/coverage/chem.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| battery | 16 | 0 | 0 | 0 | 16 | 0 |
| benchmark | 15 | 1 | 1 | 0 | 13 | 0 |
| combustion | 21 | 2 | 0 | 0 | 19 | 0 |
| electrochemistry | 6 | 0 | 0 | 0 | 6 | 0 |
| equilibrium | 6 | 0 | 0 | 0 | 6 | 0 |
| fire | 10 | 0 | 0 | 0 | 10 | 0 |
| kinetics | 18 | 2 | 1 | 0 | 15 | 0 |
| reactor | 10 | 0 | 0 | 0 | 10 | 0 |
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
| rigid-fluid | 12 | 2 | 0 | 2 | 8 | 0 |
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
| material | 16 | 3 | 0 | 1 | 12 | 0 |
| output | 12 | 2 | 0 | 0 | 10 | 0 |
| particle | 5 | 0 | 0 | 0 | 5 | 0 |
| quasi-static | 3 | 0 | 0 | 0 | 3 | 0 |
| source | 9 | 2 | 2 | 0 | 5 | 0 |
| static | 11 | 3 | 0 | 0 | 8 | 0 |
| time-domain | 16 | 6 | 0 | 3 | 7 | 0 |

## `docs/coverage/engine.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| api | 26 | 9 | 6 | 1 | 10 | 0 |
| auxiliary | 34 | 18 | 0 | 2 | 14 | 0 |
| benchmark | 8 | 3 | 2 | 0 | 3 | 0 |
| determinism | 11 | 6 | 2 | 0 | 3 | 0 |
| diagnostics | 15 | 9 | 0 | 0 | 6 | 0 |
| game-engine | 4 | 0 | 0 | 0 | 4 | 0 |
| network | 18 | 7 | 0 | 0 | 11 | 0 |
| state | 26 | 5 | 0 | 3 | 18 | 0 |
| world | 11 | 7 | 0 | 1 | 3 | 0 |

## `docs/coverage/env.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| aerodynamics | 14 | 5 | 0 | 0 | 9 | 0 |
| atmosphere | 13 | 4 | 1 | 0 | 8 | 0 |
| atmospheric-dynamics | 5 | 0 | 0 | 0 | 5 | 0 |
| benchmark | 14 | 3 | 1 | 0 | 10 | 0 |
| boundary-layer | 6 | 0 | 0 | 0 | 6 | 0 |
| cryosphere | 21 | 0 | 0 | 0 | 21 | 0 |
| environmental-load | 5 | 0 | 0 | 1 | 4 | 0 |
| moist-air | 7 | 0 | 0 | 0 | 7 | 0 |
| ocean-coastal | 5 | 0 | 0 | 0 | 5 | 0 |
| ocean-waves | 21 | 3 | 0 | 0 | 18 | 0 |
| terrain | 11 | 3 | 0 | 0 | 8 | 0 |
| turbulence-gust | 10 | 0 | 0 | 1 | 9 | 0 |
| weather | 9 | 0 | 0 | 0 | 9 | 0 |
| wind-field | 11 | 2 | 1 | 0 | 8 | 0 |

## `docs/coverage/fem.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| adaptivity | 6 | 2 | 0 | 2 | 2 | 0 |
| benchmark | 12 | 4 | 0 | 0 | 8 | 0 |
| constraint | 8 | 1 | 0 | 0 | 7 | 0 |
| contact | 6 | 0 | 0 | 0 | 5 | 1 |
| dynamics | 11 | 3 | 0 | 1 | 7 | 0 |
| eigen | 3 | 0 | 0 | 0 | 3 | 0 |
| element | 17 | 3 | 0 | 0 | 14 | 0 |
| fracture | 5 | 0 | 0 | 0 | 4 | 1 |
| integration | 6 | 1 | 0 | 0 | 4 | 1 |
| kinematics | 5 | 3 | 0 | 0 | 2 | 0 |
| linear_solver | 8 | 2 | 0 | 0 | 6 | 0 |
| load | 7 | 2 | 0 | 0 | 5 | 0 |
| material | 23 | 6 | 0 | 1 | 16 | 0 |
| nonlinear | 8 | 4 | 0 | 1 | 3 | 0 |
| output | 6 | 3 | 0 | 0 | 3 | 0 |
| tangent | 6 | 2 | 0 | 1 | 3 | 0 |

## `docs/coverage/fract.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 10 | 0 | 0 | 0 | 10 | 0 |
| crack-growth | 9 | 0 | 0 | 0 | 9 | 0 |
| csg-destruction | 9 | 6 | 1 | 1 | 1 | 0 |
| cutting | 2 | 0 | 0 | 0 | 2 | 0 |
| damage | 9 | 0 | 0 | 0 | 9 | 0 |
| discrete-crack | 13 | 0 | 0 | 0 | 13 | 0 |
| dynamic-fracture | 4 | 0 | 0 | 0 | 4 | 0 |
| effect-layer-fracture | 11 | 6 | 0 | 2 | 3 | 0 |
| erosion-wear | 16 | 3 | 1 | 1 | 11 | 0 |
| fragmentation | 9 | 0 | 0 | 0 | 9 | 0 |
| lefm | 13 | 0 | 0 | 0 | 13 | 0 |
| phase-field | 7 | 0 | 0 | 0 | 7 | 0 |

## `docs/coverage/geo.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 0 | 0 | 0 | 11 | 0 |
| elastic-waves | 11 | 0 | 1 | 0 | 10 | 0 |
| geodesy-gravity | 7 | 0 | 0 | 0 | 7 | 0 |
| geomagnetism | 7 | 0 | 0 | 0 | 7 | 0 |
| ground-motion | 10 | 0 | 1 | 0 | 9 | 0 |
| groundwater | 11 | 0 | 0 | 0 | 11 | 0 |
| hydrology | 11 | 0 | 0 | 0 | 11 | 0 |
| lithosphere | 6 | 0 | 0 | 0 | 6 | 0 |
| mantle-convection | 9 | 0 | 0 | 0 | 9 | 0 |
| ocean-circulation | 12 | 0 | 0 | 0 | 12 | 0 |
| sediment | 7 | 0 | 0 | 0 | 7 | 0 |
| seismic-numerics | 8 | 0 | 0 | 0 | 8 | 0 |
| source | 7 | 0 | 0 | 0 | 7 | 0 |
| tides | 8 | 0 | 0 | 0 | 8 | 0 |

## `docs/coverage/geotech.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| bearing-earth-pressure | 9 | 0 | 1 | 0 | 8 | 0 |
| benchmark | 9 | 0 | 0 | 0 | 9 | 0 |
| consolidation | 7 | 0 | 0 | 0 | 7 | 0 |
| constitutive | 13 | 0 | 0 | 0 | 13 | 0 |
| excavation-construction | 10 | 0 | 0 | 0 | 10 | 0 |
| frost | 5 | 0 | 0 | 0 | 5 | 0 |
| liquefaction | 6 | 0 | 0 | 0 | 6 | 0 |
| piles-ssi | 6 | 0 | 0 | 0 | 6 | 0 |
| resident-earthworks | 7 | 0 | 0 | 0 | 7 | 0 |
| slope | 9 | 0 | 1 | 0 | 8 | 0 |
| soil-state | 8 | 0 | 0 | 0 | 8 | 0 |

## `docs/coverage/mat.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 1 | 0 | 0 | 10 | 0 |
| contact material | 12 | 3 | 2 | 0 | 7 | 0 |
| damage-fracture data | 3 | 0 | 0 | 0 | 3 | 0 |
| data | 21 | 0 | 6 | 0 | 15 | 0 |
| database | 13 | 2 | 0 | 0 | 11 | 0 |
| elasticity | 12 | 3 | 1 | 1 | 7 | 0 |
| electromagnetic data | 6 | 0 | 1 | 0 | 5 | 0 |
| hyperelasticity | 14 | 6 | 1 | 0 | 7 | 0 |
| lubrication | 11 | 0 | 0 | 0 | 11 | 0 |
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
| formulation | 9 | 0 | 0 | 0 | 9 | 0 |
| mechanism | 6 | 1 | 1 | 0 | 4 | 0 |
| powertrain | 17 | 6 | 0 | 0 | 11 | 0 |
| ragdoll | 12 | 5 | 0 | 1 | 6 | 0 |
| robot-control | 5 | 0 | 0 | 1 | 4 | 0 |
| robot-dynamics | 5 | 0 | 0 | 0 | 5 | 0 |
| robot-kinematics | 11 | 0 | 0 | 0 | 11 | 0 |
| rotor | 9 | 2 | 0 | 1 | 6 | 0 |
| steering | 3 | 1 | 0 | 0 | 2 | 0 |
| suspension | 10 | 2 | 0 | 1 | 7 | 0 |
| tyre | 21 | 3 | 0 | 2 | 16 | 0 |
| vehicle | 8 | 2 | 0 | 1 | 5 | 0 |

## `docs/coverage/mfg.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 7 | 0 | 0 | 0 | 7 | 0 |
| casting | 7 | 0 | 0 | 0 | 7 | 0 |
| forming | 8 | 0 | 0 | 0 | 8 | 0 |
| injection-moulding | 7 | 0 | 0 | 0 | 7 | 0 |
| machining | 8 | 0 | 0 | 0 | 8 | 0 |
| process-chain | 2 | 0 | 0 | 0 | 2 | 0 |
| welding | 6 | 0 | 0 | 0 | 6 | 0 |

## `docs/coverage/multiphase.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 12 | 1 | 0 | 0 | 11 | 0 |
| bubble and droplet dynamics | 5 | 0 | 0 | 0 | 5 | 0 |
| combustion flow | 3 | 0 | 0 | 0 | 3 | 0 |
| interface capturing: VOF | 8 | 3 | 0 | 1 | 4 | 0 |
| interface capturing: level set | 12 | 2 | 1 | 1 | 8 | 0 |
| interface capturing: other representations | 4 | 0 | 0 | 0 | 4 | 0 |
| multiphase models (averaged / dispersed) | 12 | 0 | 0 | 0 | 12 | 0 |
| non-Newtonian phases | 2 | 0 | 0 | 0 | 2 | 0 |
| output | 2 | 0 | 0 | 0 | 2 | 0 |
| particle methods: free surface | 7 | 1 | 0 | 0 | 6 | 0 |
| phase change at the interface (flow side) | 4 | 0 | 0 | 0 | 4 | 0 |
| surface tension | 15 | 4 | 1 | 1 | 9 | 0 |
| two-phase momentum | 6 | 0 | 0 | 0 | 6 | 0 |

## `docs/coverage/nonlin.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 13 | 0 | 0 | 0 | 13 | 0 |
| bifurcation | 11 | 0 | 0 | 0 | 11 | 0 |
| cellular-automaton | 8 | 0 | 0 | 0 | 8 | 0 |
| determinism | 4 | 0 | 1 | 0 | 3 | 0 |
| hamiltonian | 7 | 0 | 0 | 0 | 7 | 0 |
| lyapunov | 6 | 0 | 0 | 0 | 6 | 0 |
| map | 6 | 0 | 0 | 0 | 6 | 0 |
| ode-system | 11 | 1 | 1 | 0 | 9 | 0 |
| pattern | 10 | 0 | 0 | 0 | 10 | 0 |
| section | 8 | 0 | 0 | 0 | 8 | 0 |
| soliton | 4 | 0 | 0 | 0 | 4 | 0 |
| synchronisation | 8 | 0 | 0 | 0 | 8 | 0 |

## `docs/coverage/nuclear.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 11 | 0 | 0 | 0 | 11 | 0 |
| charged | 9 | 0 | 0 | 0 | 9 | 0 |
| decay | 13 | 0 | 1 | 0 | 12 | 0 |
| detector | 10 | 0 | 0 | 0 | 10 | 0 |
| dose | 11 | 0 | 0 | 0 | 11 | 0 |
| foundation | 3 | 0 | 0 | 0 | 3 | 0 |
| fusion | 6 | 0 | 0 | 0 | 6 | 0 |
| neutron | 12 | 0 | 0 | 0 | 12 | 0 |
| photon | 10 | 0 | 0 | 0 | 10 | 0 |
| reactor | 17 | 0 | 0 | 0 | 17 | 0 |
| shielding | 7 | 0 | 0 | 0 | 7 | 0 |
| transport | 24 | 0 | 0 | 0 | 24 | 0 |

## `docs/coverage/num.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 10 | 3 | 0 | 0 | 7 | 0 |
| coupling | 7 | 3 | 0 | 0 | 4 | 0 |
| distributed | 6 | 0 | 0 | 0 | 6 | 0 |
| eigen | 9 | 2 | 0 | 0 | 7 | 0 |
| krylov | 10 | 4 | 0 | 1 | 5 | 0 |
| linear_direct | 7 | 1 | 0 | 0 | 6 | 0 |
| matrix_function | 2 | 0 | 0 | 0 | 2 | 0 |
| mesh | 18 | 4 | 0 | 0 | 14 | 0 |
| mesh_adaptivity | 5 | 0 | 0 | 0 | 5 | 0 |
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
| reflectance | 11 | 0 | 0 | 0 | 11 | 0 |
| scattering | 7 | 0 | 0 | 0 | 7 | 0 |
| spectral | 7 | 0 | 0 | 0 | 7 | 0 |

## `docs/coverage/orbit.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| astrophysics | 14 | 0 | 0 | 0 | 14 | 0 |
| attitude | 9 | 0 | 0 | 0 | 9 | 0 |
| benchmark | 18 | 3 | 0 | 0 | 15 | 0 |
| elements | 7 | 2 | 0 | 1 | 4 | 0 |
| frames-time | 7 | 0 | 0 | 0 | 7 | 0 |
| maneuver | 9 | 0 | 1 | 0 | 8 | 0 |
| n-body integration | 14 | 3 | 1 | 1 | 9 | 0 |
| n-body law | 7 | 4 | 0 | 1 | 2 | 0 |
| output | 3 | 0 | 1 | 0 | 2 | 0 |
| perturbation | 12 | 1 | 1 | 0 | 10 | 0 |
| propagation | 7 | 1 | 0 | 0 | 6 | 0 |
| restricted three-body | 6 | 0 | 0 | 0 | 6 | 0 |
| surface gravity | 11 | 4 | 0 | 0 | 7 | 0 |
| targeting | 5 | 0 | 0 | 0 | 5 | 0 |
| two-body | 9 | 5 | 0 | 0 | 4 | 0 |

## `docs/coverage/part.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| SPH (beyond single-phase flow) | 5 | 0 | 0 | 0 | 5 | 0 |
| analysis | 4 | 0 | 0 | 0 | 4 | 0 |
| benchmark | 14 | 2 | 1 | 0 | 11 | 0 |
| constraints | 5 | 0 | 0 | 0 | 5 | 0 |
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
| fluid | 14 | 0 | 0 | 0 | 14 | 0 |
| fusion | 6 | 0 | 0 | 0 | 6 | 0 |
| industrial | 11 | 0 | 0 | 0 | 11 | 0 |
| kinetic | 12 | 0 | 0 | 0 | 12 | 0 |
| mhd-numerics | 10 | 0 | 0 | 0 | 10 | 0 |
| parameters | 18 | 0 | 0 | 0 | 18 | 0 |
| particle-in-cell | 19 | 0 | 0 | 0 | 19 | 0 |
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
| benchmark | 20 | 0 | 0 | 0 | 20 | 0 |
| closed-form | 16 | 0 | 0 | 0 | 16 | 0 |
| foundation | 8 | 0 | 0 | 0 | 8 | 0 |
| many-body | 20 | 0 | 0 | 0 | 20 | 0 |
| open-system | 10 | 0 | 0 | 0 | 10 | 0 |
| quantum-information | 12 | 0 | 0 | 0 | 12 | 0 |
| scattering | 8 | 0 | 0 | 0 | 8 | 0 |
| semiclassical | 7 | 0 | 0 | 0 | 7 | 0 |
| solid-state | 22 | 0 | 0 | 0 | 22 | 0 |
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
| narrowphase | 18 | 12 | 2 | 2 | 2 | 0 |
| query | 7 | 3 | 0 | 2 | 2 | 0 |
| sleep | 3 | 3 | 0 | 0 | 0 | 0 |
| solver | 15 | 7 | 0 | 1 | 7 | 0 |

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

## `docs/coverage/shock.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| ale-hydrocode | 6 | 0 | 0 | 0 | 6 | 0 |
| benchmark | 19 | 7 | 5 | 0 | 7 | 0 |
| boundary | 7 | 3 | 0 | 0 | 4 | 0 |
| closed-form | 15 | 4 | 0 | 0 | 11 | 0 |
| detonation | 5 | 0 | 0 | 0 | 5 | 0 |
| eos | 7 | 1 | 0 | 0 | 6 | 0 |
| governing | 8 | 1 | 0 | 0 | 7 | 0 |
| multi-dimensional | 5 | 0 | 0 | 0 | 5 | 0 |
| output | 2 | 0 | 0 | 0 | 2 | 0 |
| positivity-entropy | 10 | 5 | 1 | 2 | 2 | 0 |
| reconstruction | 10 | 3 | 1 | 0 | 6 | 0 |
| riemann | 14 | 2 | 0 | 0 | 12 | 0 |
| time | 7 | 3 | 0 | 0 | 4 | 0 |

## `docs/coverage/soft.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| attachment / external force | 6 | 1 | 0 | 2 | 3 | 0 |
| benchmark | 8 | 3 | 1 | 0 | 4 | 0 |
| cloth model | 14 | 2 | 0 | 0 | 12 | 0 |
| collision | 16 | 4 | 0 | 2 | 10 | 0 |
| integration / output | 4 | 2 | 0 | 0 | 2 | 0 |
| rod / rope / hair | 15 | 3 | 0 | 1 | 11 | 0 |
| soft body | 9 | 0 | 3 | 1 | 5 | 0 |
| solver | 15 | 3 | 0 | 0 | 12 | 0 |
| time integration / damping | 4 | 2 | 0 | 0 | 2 | 0 |
| topology change | 5 | 1 | 0 | 0 | 4 | 0 |

## `docs/coverage/stat.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 13 | 0 | 0 | 0 | 13 | 0 |
| critical | 9 | 0 | 0 | 0 | 9 | 0 |
| estimator | 7 | 0 | 0 | 0 | 7 | 0 |
| free-energy | 6 | 0 | 0 | 0 | 6 | 0 |
| kinetic-mc | 7 | 0 | 0 | 0 | 7 | 0 |
| lattice-model | 13 | 0 | 0 | 0 | 13 | 0 |
| mc-advanced | 9 | 0 | 0 | 0 | 9 | 0 |
| mc-local | 10 | 0 | 0 | 0 | 10 | 0 |
| md-ensemble | 6 | 0 | 0 | 0 | 6 | 0 |
| nonequilibrium | 7 | 0 | 0 | 0 | 7 | 0 |
| sequence | 4 | 0 | 0 | 0 | 4 | 0 |
| stochastic-dynamics | 12 | 0 | 0 | 0 | 12 | 0 |
| transport | 4 | 0 | 0 | 0 | 4 | 0 |

## `docs/coverage/struct.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| beam | 14 | 2 | 0 | 0 | 12 | 0 |
| buckling | 16 | 3 | 0 | 1 | 12 | 0 |
| contact-stress | 3 | 1 | 0 | 0 | 2 | 0 |
| creep | 9 | 1 | 0 | 2 | 6 | 0 |
| fatigue | 17 | 3 | 0 | 2 | 12 | 0 |
| joint | 18 | 4 | 0 | 1 | 13 | 0 |
| laminate | 18 | 8 | 0 | 0 | 10 | 0 |
| plate-shell | 8 | 0 | 0 | 0 | 8 | 0 |
| section | 10 | 2 | 0 | 0 | 8 | 0 |
| stress-concentration | 8 | 2 | 0 | 2 | 4 | 0 |
| synthesis | 3 | 0 | 0 | 1 | 2 | 0 |
| thermal-stress | 4 | 1 | 0 | 1 | 2 | 0 |
| torsion | 4 | 0 | 0 | 0 | 4 | 0 |
| vibration | 20 | 6 | 0 | 0 | 14 | 0 |

## `docs/coverage/therm.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 15 | 2 | 0 | 0 | 13 | 0 |
| boundary condition | 10 | 1 | 0 | 0 | 9 | 0 |
| closed-form conduction | 10 | 1 | 0 | 0 | 9 | 0 |
| conduction | 18 | 8 | 0 | 1 | 9 | 0 |
| convection (closed form) | 12 | 0 | 0 | 0 | 12 | 0 |
| coupling (thermal side) | 6 | 3 | 0 | 0 | 3 | 0 |
| fin / heat exchanger | 5 | 0 | 0 | 0 | 5 | 0 |
| heat source | 8 | 1 | 0 | 1 | 6 | 0 |
| output | 3 | 1 | 0 | 0 | 2 | 0 |
| phase change | 9 | 0 | 0 | 1 | 8 | 0 |
| radiation | 15 | 0 | 0 | 0 | 15 | 0 |
| thermodynamics | 17 | 0 | 0 | 0 | 17 | 0 |
