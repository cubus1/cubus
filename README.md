# cubus

CogRecord architecture, cognitive pipeline, oracle, and ghost discovery.

Depends on [numrus](https://github.com/cubus1/numrus) for SIMD-accelerated numerics.

## Crates

| Crate | Description |
|-------|-------------|
| cubus | CogRecord v3 format: phase-space, carrier waveforms, focus gating, holograph |
| cubus-lance | Arrow/Lance storage bridge for CogRecord persistence |
| cubus-oracle | Three-temperature holographic oracle with capacity sweep |
| cubus-ghost | Ghost discovery: emergent connections in holographic containers |

## Dependency Graph

```
cubus ──→ numrus-core, numrus-clam
cubus-lance ──→ numrus-core, numrus-clam
cubus-oracle ──→ numrus-core, numrus-nars, numrus-substrate, cubus
cubus-ghost ──→ numrus-core, numrus-nars, numrus-substrate
```
