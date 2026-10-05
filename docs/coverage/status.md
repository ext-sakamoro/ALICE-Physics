# Coverage status

Generated from docs/coverage/*.toml by `python3 scripts/coverage_check.py --write-status`;
do not edit by hand. Each table lists the capabilities a field expects, with what the
crate has today.

| table | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| `docs/coverage/cfd.toml` | 117 | 39 | 2 | 6 | 70 | 0 |
| `docs/coverage/fem.toml` | 133 | 36 | 0 | 6 | 88 | 3 |
| `docs/coverage/multiphase.toml` | 86 | 11 | 2 | 3 | 70 | 0 |
| `docs/coverage/num.toml` | 101 | 22 | 1 | 5 | 72 | 1 |
| `docs/coverage/part.toml` | 119 | 26 | 2 | 3 | 88 | 0 |
| `docs/coverage/soft.toml` | 90 | 21 | 4 | 6 | 59 | 0 |
| `docs/coverage/struct.toml` | 134 | 33 | 0 | 10 | 91 | 0 |
| `docs/coverage/therm.toml` | 102 | 17 | 0 | 3 | 82 | 0 |
| **total** | 882 | 205 | 11 | 42 | 620 | 4 |

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
| stochastic | 2 | 0 | 1 | 0 | 1 | 0 |
| time_integration | 14 | 0 | 0 | 0 | 14 | 0 |
| transform | 2 | 0 | 0 | 0 | 2 | 0 |

## `docs/coverage/part.toml`

| axis | items | implemented+oracle | implemented-no-oracle | partial | missing | out-of-scope |
|---|---:|---:|---:|---:|---:|---:|
| SPH (beyond single-phase flow) | 5 | 0 | 0 | 0 | 5 | 0 |
| analysis | 4 | 0 | 0 | 0 | 4 | 0 |
| benchmark | 14 | 2 | 1 | 0 | 11 | 0 |
| constraints | 3 | 0 | 0 | 0 | 3 | 0 |
| crowd | 9 | 5 | 0 | 0 | 4 | 0 |
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
