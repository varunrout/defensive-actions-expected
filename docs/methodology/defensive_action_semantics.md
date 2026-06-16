# Defensive action semantics

The current modelling table includes recorded defensive interventions only when a defensible actor/possession interpretation is available. The model estimates short-horizon attacking threat following recorded defensive actions; it is not a causal DAx or invisible-defending model.

| Event family | Included? | Actor interpretation | Possession context | Notes |
|---|---:|---|---|---|
| Pressure | Yes | Pressing player | Usually opponent possession | Active pressure; does not necessarily end possession. |
| Ball Recovery | Yes | Recovering player | Previous possession must be checked | Can start new possession; use previous attacking context where available. |
| Duel | Yes | Contesting player | Context-dependent | Contest event; not automatically a second-ball tactical phase. |
| 50/50 | Yes | Contesting player | Context-dependent | Similar to duel. |
| Clearance | Yes | Clearing defender | Usually defending under pressure | Often possession-ending; evaluate pre-action attacking possession. |
| Block | Yes | Blocking defender | Usually opponent attack | Intervention near shot/pass lane. |
| Interception | Yes | Intercepting defender | Opponent possession before action | May win possession. |
| Foul Committed | Yes | Fouling defender | Opponent possession often relevant | Discipline/interruption, not necessarily successful defending. |
| Goal Keeper | No by default | Mixed goalkeeper actions | Mixed | Generic goalkeeper events include distributions/possession actions and are excluded unless subtyped later. |
| Shield | No | Possession protection | Actor may be in possession | Excluded because current audit does not prove defensive interpretation. |

Required model-facing semantics: `actor_team`, `attacking_team`, and `defending_team` are separate and must not be overwritten by assumptions.
