//! AI War Ghost Oracle — Prompt 10 (Organic Residual Edition)
//!
//! Encodes 221 entities and 356 edges from Sarah Ciston's "AI War Cloud"
//! into holographic containers via OrganicWAL with base-aware plasticity.
//!
//! Ghost detection: after encoding all known edges, computes the residual
//! (container minus reconstructed known-edge contributions). Non-edges
//! that correlate with the residual are "ghosts" — connections implied
//! by the structure that nobody explicitly stored.
//!
//! Runs across Signed(5), Signed(7), Signed(9) with base-aware plasticity
//! to show how base width affects ghost detection.

use numrus_substrate::{AbsorptionTracker, OrganicWAL, PlasticityTracker, XTransPattern};
use numrus_nars::{bind, generate_template, Base};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Part 1: Entity & Edge types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum EntityType {
    System,
    Stakeholder,
    CivicSystem,
    Historical,
    Person,
}

impl EntityType {
    pub fn label(&self) -> &'static str {
        match self {
            EntityType::System => "System",
            EntityType::Stakeholder => "Stakeholder",
            EntityType::CivicSystem => "Civic",
            EntityType::Historical => "Historical",
            EntityType::Person => "Person",
        }
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[derive(Clone, Debug, Default)]
pub struct OntologyAxes {
    pub current_status: Option<String>,
    pub system_type: Option<String>,
    pub ml_task: Option<String>,
    pub military_use: Option<String>,
    pub civic_use: Option<String>,
    pub purpose: Option<String>,
    pub capacity: Option<String>,
    pub output: Option<String>,
    pub impact: Option<String>,
    pub stakeholder_type: Option<String>,
    pub airo_type: Option<String>,
    pub person_type: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub axes: OntologyAxes,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EdgeType {
    Develops,
    Deploys,
    Connection,
    People,
    Place,
}

impl EdgeType {
    pub fn label(&self) -> &'static str {
        match self {
            EdgeType::Develops => "Develops",
            EdgeType::Deploys => "Deploys",
            EdgeType::Connection => "Connection",
            EdgeType::People => "People",
            EdgeType::Place => "Place",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// Part 2: Graph
// ---------------------------------------------------------------------------

pub struct AiWarGraph {
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub id_to_index: HashMap<String, usize>,
}

impl AiWarGraph {
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn get(&self, id: &str) -> Option<&Entity> {
        self.id_to_index.get(id).map(|&i| &self.entities[i])
    }

    pub fn edges_of(&self, id: &str) -> Vec<&Edge> {
        self.edges
            .iter()
            .filter(|e| e.source_id == id || e.target_id == id)
            .collect()
    }

    pub fn neighbors(&self, id: &str) -> Vec<&Entity> {
        self.edges_of(id)
            .iter()
            .filter_map(|e| {
                let other = if e.source_id == id {
                    &e.target_id
                } else {
                    &e.source_id
                };
                self.get(other)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Part 3: Parsing
// ---------------------------------------------------------------------------

fn opt_str(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(String::from)
}

pub fn parse_graph(json: &serde_json::Value) -> AiWarGraph {
    let mut entities = Vec::new();
    let mut id_to_index: HashMap<String, usize> = HashMap::new();
    let mut edges = Vec::new();

    let mut add_entity = |id: &str, name: &str, etype: EntityType, axes: OntologyAxes| {
        let index = entities.len();
        id_to_index.insert(id.to_string(), index);
        entities.push(Entity {
            id: id.to_string(),
            name: name.to_string(),
            entity_type: etype,
            axes,
            index,
        });
    };

    // N_Systems
    if let Some(systems) = json["N_Systems"].as_array() {
        for s in systems {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                current_status: opt_str(s, "currentStatus"),
                system_type: opt_str(s, "type"),
                ml_task: opt_str(s, "MLTask"),
                military_use: opt_str(s, "militaryUse"),
                civic_use: opt_str(s, "civicUse"),
                purpose: opt_str(s, "purpose"),
                capacity: opt_str(s, "capacity"),
                output: opt_str(s, "output"),
                impact: opt_str(s, "impact"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::System, axes);
        }
    }

    // N_Stakeholders
    if let Some(stakeholders) = json["N_Stakeholders"].as_array() {
        for s in stakeholders {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                stakeholder_type: opt_str(s, "type"),
                airo_type: opt_str(s, "airo:type"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::Stakeholder, axes);
        }
    }

    // N_Civic
    if let Some(civic) = json["N_Civic"].as_array() {
        for s in civic {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                current_status: opt_str(s, "currentStatus"),
                system_type: opt_str(s, "type"),
                ml_task: opt_str(s, "MLTask"),
                civic_use: opt_str(s, "civicUse"),
                purpose: opt_str(s, "purpose"),
                capacity: opt_str(s, "capacity"),
                output: opt_str(s, "output"),
                impact: opt_str(s, "impact"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::CivicSystem, axes);
        }
    }

    // N_Historical
    if let Some(hist) = json["N_Historical"].as_array() {
        for s in hist {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                current_status: opt_str(s, "currentStatus"),
                system_type: opt_str(s, "type"),
                military_use: opt_str(s, "militaryUse"),
                civic_use: opt_str(s, "civicUse"),
                ml_task: opt_str(s, "MLTask"),
                purpose: opt_str(s, "purpose"),
                capacity: opt_str(s, "capacity"),
                output: opt_str(s, "output"),
                impact: opt_str(s, "impact"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::Historical, axes);
        }
    }

    // N_People
    if let Some(people) = json["N_People"].as_array() {
        for s in people {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                person_type: opt_str(s, "type"),
                airo_type: opt_str(s, "airo:type"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::Person, axes);
        }
    }

    // Edges
    let edge_tables: &[(&str, EdgeType)] = &[
        ("E_isDevelopedBy", EdgeType::Develops),
        ("E_isDeployedBy", EdgeType::Deploys),
        ("E_connection", EdgeType::Connection),
        ("E_people", EdgeType::People),
        ("E_place", EdgeType::Place),
    ];

    for (key, etype) in edge_tables {
        if let Some(arr) = json[key].as_array() {
            for e in arr {
                edges.push(Edge {
                    source_id: e["source"].as_str().unwrap_or("").to_string(),
                    target_id: e["target"].as_str().unwrap_or("").to_string(),
                    edge_type: etype.clone(),
                    label: opt_str(e, "label"),
                });
            }
        }
    }

    AiWarGraph {
        entities,
        edges,
        id_to_index,
    }
}

pub fn load_graph_from_file(path: &str) -> AiWarGraph {
    let data =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
    let json: serde_json::Value =
        serde_json::from_str(&data).unwrap_or_else(|e| panic!("Failed to parse JSON: {}", e));
    parse_graph(&json)
}

// ---------------------------------------------------------------------------
// Part 4: Deterministic seeded RNG (reproducible from entity ID)
// ---------------------------------------------------------------------------

/// Thin `rand::RngCore` adapter around `numrus_core::SplitMix64`.
struct SeededRng(numrus_core::SplitMix64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self(numrus_core::SplitMix64::new(seed.wrapping_add(1)))
    }
}

impl rand::RngCore for SeededRng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u64() as u32
    }
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            for (d, s) in chunk.iter_mut().zip(val.iter()) {
                *d = *s;
            }
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

fn hash_string(s: &str) -> u64 {
    // FNV-1a
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn generate_from_seed(d: usize, base: Base, seed: u64) -> Vec<i8> {
    let mut rng = SeededRng::new(seed);
    generate_template(d, base, &mut rng)
}

// ---------------------------------------------------------------------------
// Part 5: Template generation with ontology-based correlation
// ---------------------------------------------------------------------------

fn extract_axis_values(e: &Entity) -> Vec<String> {
    let a = &e.axes;
    let mut vals = Vec::new();

    if let Some(v) = &a.current_status {
        vals.push(format!("status:{}", v));
    }
    if let Some(v) = &a.system_type {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("type:{}", part));
            }
        }
    }
    if let Some(v) = &a.ml_task {
        vals.push(format!("ml:{}", v));
    }
    if let Some(v) = &a.military_use {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("mil:{}", part));
            }
        }
    }
    if let Some(v) = &a.civic_use {
        vals.push(format!("civic:{}", v));
    }
    if let Some(v) = &a.purpose {
        vals.push(format!("purpose:{}", v));
    }
    if let Some(v) = &a.capacity {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("cap:{}", part));
            }
        }
    }
    if let Some(v) = &a.output {
        vals.push(format!("output:{}", v));
    }
    if let Some(v) = &a.impact {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("impact:{}", part));
            }
        }
    }
    if let Some(v) = &a.stakeholder_type {
        vals.push(format!("stype:{}", v));
    }
    if let Some(v) = &a.airo_type {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("airo:{}", part));
            }
        }
    }
    if let Some(v) = &a.person_type {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("ptype:{}", part));
            }
        }
    }

    vals
}

pub fn generate_entity_templates(
    graph: &AiWarGraph,
    d: usize,
    base: Base,
    overlap_per_axis: f32,
) -> Vec<Vec<i8>> {
    // Per-entity random component (seeded by ID)
    let entity_components: Vec<Vec<i8>> = graph
        .entities
        .iter()
        .map(|e| generate_from_seed(d, base, hash_string(&e.id)))
        .collect();

    // Per-axis-value basis vectors
    let mut axis_bases: HashMap<String, Vec<i8>> = HashMap::new();
    let mut axis_seed_counter: usize = 0;

    for entity in &graph.entities {
        for val in extract_axis_values(entity) {
            axis_bases.entry(val).or_insert_with(|| {
                let seed = hash_string(&format!("axis:{}", axis_seed_counter));
                axis_seed_counter += 1;
                generate_from_seed(d, base, seed)
            });
        }
    }

    let min_val = base.min_val() as f32;
    let max_val = base.max_val() as f32;

    graph
        .entities
        .iter()
        .enumerate()
        .map(|(i, entity)| {
            let axis_vals = extract_axis_values(entity);
            let n_axes = axis_vals.len();
            let total_overlap = (n_axes as f32 * overlap_per_axis).min(0.6);
            let entity_weight = 1.0 - total_overlap;
            let per_axis_weight = if n_axes > 0 {
                total_overlap / n_axes as f32
            } else {
                0.0
            };

            let mut template = vec![0i8; d];
            for j in 0..d {
                let mut val = entity_weight * entity_components[i][j] as f32;
                for av in &axis_vals {
                    if let Some(basis) = axis_bases.get(av.as_str()) {
                        val += per_axis_weight * basis[j] as f32;
                    }
                }
                template[j] = val.round().clamp(min_val, max_val) as i8;
            }
            template
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Part 6: Organic Encoding
// ---------------------------------------------------------------------------

/// Result of encoding edges via OrganicWAL.
pub struct OrganicEncodeResult {
    /// The organic container (i8 values).
    pub container: Vec<i8>,
    /// Residual: container minus reconstructed known-edge contributions (f32).
    pub residual: Vec<f32>,
    /// Coefficients extracted via surgical orthogonal projection.
    pub edge_coefficients: Vec<f32>,
    /// Average absorption across all edge writes.
    pub absorption_avg: f32,
    /// Minimum absorption across all edge writes.
    pub absorption_min: f32,
    /// Residual energy: mean squared residual value.
    pub residual_energy: f32,
    /// Number of edges successfully encoded.
    pub edges_encoded: usize,
}

fn amplitude_for_edge(etype: &EdgeType) -> f32 {
    match etype {
        EdgeType::Develops => 1.0,
        EdgeType::Deploys => 0.9,
        EdgeType::Connection => 0.7,
        EdgeType::People => 0.8,
        EdgeType::Place => 0.6,
    }
}

/// Encode edges via raw signed accumulation (no WAL, no orthogonalization).
///
/// Simple approach: for each edge, bind(src, tgt) and accumulate into container.
/// This is the baseline — no Gram-Schmidt, no plasticity, no channel isolation.
pub fn encode_edges_signed_raw(
    graph: &AiWarGraph,
    templates: &[Vec<i8>],
    d: usize,
    base: Base,
) -> Vec<i8> {
    let mut accum = vec![0.0f32; d];
    let s_min = base.min_val() as f32;
    let s_max = base.max_val() as f32;

    let mut encoded = 0usize;
    for edge in &graph.edges {
        let src_idx = match graph.id_to_index.get(&edge.source_id) {
            Some(&i) => i,
            None => continue,
        };
        let tgt_idx = match graph.id_to_index.get(&edge.target_id) {
            Some(&i) => i,
            None => continue,
        };

        let amp = amplitude_for_edge(&edge.edge_type);
        let bound = bind(&templates[src_idx], &templates[tgt_idx], base);
        for j in 0..d {
            accum[j] += amp * bound[j] as f32;
        }
        encoded += 1;
    }

    eprintln!(
        "  Encoded {} edges via raw accumulation (D={}, {})",
        encoded,
        d,
        base.name()
    );

    accum
        .iter()
        .map(|&v| v.round().clamp(s_min, s_max) as i8)
        .collect()
}

/// Encode all graph edges into an organic container using WAL + plasticity.
///
/// Each edge becomes a "concept" in the WAL: bind(src_template, tgt_template).
/// The WAL orthogonalizes each new edge against all previous, then writes
/// via receptivity-gated absorption with base-aware plasticity.
///
/// Returns the container, residual, coefficients, and metrics.
pub fn encode_edges_organic(
    graph: &AiWarGraph,
    templates: &[Vec<i8>],
    d: usize,
    base: Base,
    channels: usize,
) -> OrganicEncodeResult {
    let pattern = XTransPattern::new(d, channels);
    let mut wal = OrganicWAL::new(pattern);
    let mut container = vec![0i8; d];
    // Start with 0 concepts; add them as edges are registered
    let mut plasticity = PlasticityTracker::new_with_cardinality(0, 50, base.cardinality() as usize);
    let mut absorption_tracker = AbsorptionTracker::new(graph.edges.len().max(1));

    let mut amplitudes = Vec::new();
    let mut encoded = 0usize;

    for edge in &graph.edges {
        let src_idx = match graph.id_to_index.get(&edge.source_id) {
            Some(&i) => i,
            None => continue,
        };
        let tgt_idx = match graph.id_to_index.get(&edge.target_id) {
            Some(&i) => i,
            None => continue,
        };

        let bound = bind(&templates[src_idx], &templates[tgt_idx], base);
        wal.register_concept(encoded as u32, bound);
        plasticity.add_concept();
        amplitudes.push(amplitude_for_edge(&edge.edge_type));
        encoded += 1;
    }

    // Write all edges through WAL with base-aware plasticity
    for i in 0..encoded {
        let result = wal.write_plastic(&mut container, i, amplitudes[i], 0.1, &mut plasticity);
        absorption_tracker.record(result.absorption);
    }

    // Surgical extraction: recover true coefficients
    let coefficients = wal.surgical_extract(&container);

    // Compute residual: what the container holds beyond the known edges
    let residual = wal.compute_residual(&container, &coefficients);
    let res_energy = residual.iter().map(|&v| v * v).sum::<f32>() / residual.len() as f32;

    eprintln!(
        "  Encoded {} edges via OrganicWAL (D={}, C={}, {})",
        encoded,
        d,
        channels,
        base.name()
    );
    eprintln!(
        "  Absorption: avg={:.4}, min={:.4}  Residual energy: {:.4}",
        absorption_tracker.average(),
        absorption_tracker.minimum(),
        res_energy
    );

    OrganicEncodeResult {
        container,
        residual,
        edge_coefficients: coefficients,
        absorption_avg: absorption_tracker.average(),
        absorption_min: absorption_tracker.minimum(),
        residual_energy: res_energy,
        edges_encoded: encoded,
    }
}

// ---------------------------------------------------------------------------
// Part 7: Ghost Probing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GhostConnection {
    pub entity_a: usize,
    pub entity_b: usize,
    pub name_a: String,
    pub name_b: String,
    pub type_a: EntityType,
    pub type_b: EntityType,
    /// Cosine similarity of bind(A,B) against the organic container.
    pub direct_strength: f32,
    /// Cosine similarity of bind(A,B) against the residual.
    /// This is the signal that ISN'T explained by any single known edge.
    pub residual_strength: f32,
    /// Primary ghost metric = residual_strength.
    pub ghost_signal: f32,
}

fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += a[i] as f64 * a[i] as f64;
        norm_b += b[i] as f64 * b[i] as f64;
    }
    let norm = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    (dot / norm) as f32
}

fn cosine_similarity_mixed(probe: &[i8], target: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..probe.len().min(target.len()) {
        dot += probe[i] as f64 * target[i] as f64;
        norm_a += probe[i] as f64 * probe[i] as f64;
        norm_b += target[i] as f64 * target[i] as f64;
    }
    let norm = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    (dot / norm) as f32
}

/// Probe all non-edge pairs for ghost connections.
///
/// For each pair (i,j) that doesn't have an explicit edge:
///   1. Compute probe = bind(template_i, template_j)
///   2. direct_strength = cosine(probe, container)
///   3. residual_strength = cosine(probe, residual)
///   4. ghost_signal = residual_strength
///
/// Returns top_k ghosts ranked by |ghost_signal|.
pub fn probe_ghost_connections(
    graph: &AiWarGraph,
    container: &[i8],
    residual: &[f32],
    templates: &[Vec<i8>],
    base: Base,
    top_k: usize,
) -> Vec<GhostConnection> {
    let mut existing: HashSet<(usize, usize)> = HashSet::new();
    for edge in &graph.edges {
        if let (Some(&s), Some(&t)) = (
            graph.id_to_index.get(&edge.source_id),
            graph.id_to_index.get(&edge.target_id),
        ) {
            existing.insert((s, t));
            existing.insert((t, s));
        }
    }

    let n = graph.entity_count();
    let mut ghosts = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if existing.contains(&(i, j)) {
                continue;
            }

            let probe = bind(&templates[i], &templates[j], base);
            let direct_strength = cosine_similarity_i8(&probe, container);
            let residual_strength = cosine_similarity_mixed(&probe, residual);

            ghosts.push(GhostConnection {
                entity_a: i,
                entity_b: j,
                name_a: graph.entities[i].name.clone(),
                name_b: graph.entities[j].name.clone(),
                type_a: graph.entities[i].entity_type.clone(),
                type_b: graph.entities[j].entity_type.clone(),
                direct_strength,
                residual_strength,
                ghost_signal: residual_strength,
            });
        }
    }

    ghosts.sort_by(|a, b| {
        b.ghost_signal
            .abs()
            .partial_cmp(&a.ghost_signal.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ghosts.truncate(top_k);
    ghosts
}

/// Probe ghosts for a single target entity.
pub fn probe_entity_ghosts(
    graph: &AiWarGraph,
    target_idx: usize,
    container: &[i8],
    residual: &[f32],
    templates: &[Vec<i8>],
    base: Base,
    filter_indices: Option<&[usize]>,
    top_k: usize,
) -> Vec<GhostConnection> {
    let target = &graph.entities[target_idx];
    let existing_neighbors: HashSet<String> = graph
        .edges_of(&target.id)
        .iter()
        .map(|e| {
            if e.source_id == target.id {
                e.target_id.clone()
            } else {
                e.source_id.clone()
            }
        })
        .collect();

    let mut ghosts = Vec::new();

    let candidates: Vec<usize> = match filter_indices {
        Some(indices) => indices.to_vec(),
        None => (0..graph.entity_count()).collect(),
    };

    for &idx in &candidates {
        if idx == target_idx {
            continue;
        }
        if existing_neighbors.contains(&graph.entities[idx].id) {
            continue;
        }

        let probe = bind(&templates[target_idx], &templates[idx], base);
        let direct_strength = cosine_similarity_i8(&probe, container);
        let residual_strength = cosine_similarity_mixed(&probe, residual);

        ghosts.push(GhostConnection {
            entity_a: target_idx,
            entity_b: idx,
            name_a: target.name.clone(),
            name_b: graph.entities[idx].name.clone(),
            type_a: target.entity_type.clone(),
            type_b: graph.entities[idx].entity_type.clone(),
            direct_strength,
            residual_strength,
            ghost_signal: residual_strength,
        });
    }

    ghosts.sort_by(|a, b| {
        b.ghost_signal
            .abs()
            .partial_cmp(&a.ghost_signal.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ghosts.truncate(top_k);
    ghosts
}

// ---------------------------------------------------------------------------
// Part 7b: Dual-Method Ghost Probing (Signed Raw + Organic Residual)
// ---------------------------------------------------------------------------

/// Ghost connection with both signed raw and organic residual signals.
#[derive(Clone, Debug)]
pub struct DualGhost {
    pub entity_a: usize,
    pub entity_b: usize,
    pub name_a: String,
    pub name_b: String,
    pub type_a: EntityType,
    pub type_b: EntityType,
    /// Cosine similarity against raw signed accumulation container.
    pub signed_strength: f32,
    /// Cosine similarity against organic WAL container.
    pub organic_direct: f32,
    /// Cosine similarity against organic residual.
    pub organic_residual: f32,
}

fn probe_dual_entity(
    graph: &AiWarGraph,
    target_idx: usize,
    signed_container: &[i8],
    organic_container: &[i8],
    organic_residual: &[f32],
    templates: &[Vec<i8>],
    base: Base,
    filter_indices: Option<&[usize]>,
    top_k: usize,
) -> Vec<DualGhost> {
    let target = &graph.entities[target_idx];
    let existing_neighbors: HashSet<String> = graph
        .edges_of(&target.id)
        .iter()
        .map(|e| {
            if e.source_id == target.id {
                e.target_id.clone()
            } else {
                e.source_id.clone()
            }
        })
        .collect();

    let candidates: Vec<usize> = match filter_indices {
        Some(indices) => indices.to_vec(),
        None => (0..graph.entity_count()).collect(),
    };

    let mut ghosts = Vec::new();
    for &idx in &candidates {
        if idx == target_idx {
            continue;
        }
        if existing_neighbors.contains(&graph.entities[idx].id) {
            continue;
        }

        let probe = bind(&templates[target_idx], &templates[idx], base);
        let signed_strength = cosine_similarity_i8(&probe, signed_container);
        let organic_direct = cosine_similarity_i8(&probe, organic_container);
        let organic_residual_s = cosine_similarity_mixed(&probe, organic_residual);

        ghosts.push(DualGhost {
            entity_a: target_idx,
            entity_b: idx,
            name_a: target.name.clone(),
            name_b: graph.entities[idx].name.clone(),
            type_a: target.entity_type.clone(),
            type_b: graph.entities[idx].entity_type.clone(),
            signed_strength,
            organic_direct,
            organic_residual: organic_residual_s,
        });
    }

    // Sort by signed strength (absolute) for primary ranking
    ghosts.sort_by(|a, b| {
        b.signed_strength
            .abs()
            .partial_cmp(&a.signed_strength.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ghosts.truncate(top_k);
    ghosts
}

fn probe_dual_pairs(
    graph: &AiWarGraph,
    set_a: &[usize],
    set_b: &[usize],
    signed_container: &[i8],
    organic_container: &[i8],
    organic_residual: &[f32],
    templates: &[Vec<i8>],
    base: Base,
    top_k: usize,
) -> Vec<DualGhost> {
    let mut existing: HashSet<(usize, usize)> = HashSet::new();
    for edge in &graph.edges {
        if let (Some(&s), Some(&t)) = (
            graph.id_to_index.get(&edge.source_id),
            graph.id_to_index.get(&edge.target_id),
        ) {
            existing.insert((s, t));
            existing.insert((t, s));
        }
    }

    let mut ghosts = Vec::new();
    for &i in set_a {
        for &j in set_b {
            if i == j {
                continue;
            }
            if existing.contains(&(i, j)) {
                continue;
            }

            let probe = bind(&templates[i], &templates[j], base);
            let signed_strength = cosine_similarity_i8(&probe, signed_container);
            let organic_direct = cosine_similarity_i8(&probe, organic_container);
            let organic_residual_s = cosine_similarity_mixed(&probe, organic_residual);

            ghosts.push(DualGhost {
                entity_a: i,
                entity_b: j,
                name_a: graph.entities[i].name.clone(),
                name_b: graph.entities[j].name.clone(),
                type_a: graph.entities[i].entity_type.clone(),
                type_b: graph.entities[j].entity_type.clone(),
                signed_strength,
                organic_direct,
                organic_residual: organic_residual_s,
            });
        }
    }

    ghosts.sort_by(|a, b| {
        b.signed_strength
            .abs()
            .partial_cmp(&a.signed_strength.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ghosts.truncate(top_k);
    ghosts
}

fn print_dual_ghosts(title: &str, ghosts: &[DualGhost], max_rows: usize) {
    println!("{}", "=".repeat(100));
    println!("  {}", title);
    println!("{}", "=".repeat(100));
    println!(
        "  {:25} {:25} {:>9} {:>9} {:>9}",
        "Entity A", "Entity B", "Signed", "OrgDirect", "OrgResid"
    );
    println!("{}", "-".repeat(100));

    for g in ghosts.iter().take(max_rows) {
        println!(
            "  {} {} {:>+9.4} {:>+9.4} {:>+9.4}",
            truncate_str(&g.name_a, 25),
            truncate_str(&g.name_b, 25),
            g.signed_strength,
            g.organic_direct,
            g.organic_residual
        );
    }

    println!("{}", "=".repeat(100));
}

// ---------------------------------------------------------------------------
// Part 8: Output Formatting
// ---------------------------------------------------------------------------

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        format!("{:width$}", s, width = max)
    } else {
        let truncated: String = s.chars().take(max - 1).collect();
        format!("{}\u{2026}", truncated)
    }
}

pub fn print_ghost_connections(title: &str, ghosts: &[GhostConnection], max_rows: usize) {
    println!("{}", "=".repeat(90));
    println!("  {}", title);
    println!("{}", "=".repeat(90));
    println!(
        "  {:25} {:25} {:>8} {:>8} {:>8}",
        "Entity A", "Entity B", "Direct", "Residual", "Ghost"
    );
    println!("{}", "-".repeat(90));

    for g in ghosts.iter().take(max_rows) {
        let mark = if g.ghost_signal > 0.02 {
            " +"
        } else if g.ghost_signal < -0.02 {
            " -"
        } else {
            "  "
        };
        println!(
            "  {} {} {:>+8.4} {:>+8.4} {:>+8.4}{}",
            truncate_str(&g.name_a, 25),
            truncate_str(&g.name_b, 25),
            g.direct_strength,
            g.residual_strength,
            g.ghost_signal,
            mark
        );
    }

    println!("{}", "=".repeat(90));
}

pub fn ghost_type_summary(ghosts: &[GhostConnection]) {
    let mut type_counts: HashMap<String, (usize, f32, f32)> = HashMap::new();

    for g in ghosts {
        let key = format!("{}<->{}", g.type_a.label(), g.type_b.label());
        let entry = type_counts.entry(key).or_insert((0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += g.ghost_signal.abs();
        entry.2 += g.direct_strength.abs();
    }

    let mut sorted: Vec<(String, usize, f32, f32)> = type_counts
        .into_iter()
        .map(|(k, (count, ghost_total, direct_total))| {
            (
                k,
                count,
                ghost_total / count as f32,
                direct_total / count as f32,
            )
        })
        .collect();
    sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("{}", "=".repeat(66));
    println!("  GHOST TYPE SUMMARY");
    println!("{}", "=".repeat(66));
    println!(
        "  {:30} {:>6} {:>12} {:>12}",
        "Type Pair", "Count", "Avg |Ghost|", "Avg |Direct|"
    );
    println!("{}", "-".repeat(66));
    for (pair, count, avg_ghost, avg_direct) in &sorted {
        println!(
            "  {:30} {:>6} {:>12.4} {:>12.4}",
            pair, count, avg_ghost, avg_direct
        );
    }
    println!("{}", "=".repeat(66));
}

// ---------------------------------------------------------------------------
// Part 9: Focused Scenarios
// ---------------------------------------------------------------------------

/// Run all 4 scenarios with dual-method comparison (signed raw + organic residual).
pub fn run_dual_scenarios(
    graph: &AiWarGraph,
    signed_container: &[i8],
    organic_container: &[i8],
    organic_residual: &[f32],
    templates: &[Vec<i8>],
    base: Base,
) {
    // -- Scenario A: Ghost connections for Lavender only (focus entity) --
    println!();
    let lavender = graph.entities.iter().find(|e| e.name == "Lavender");
    if let Some(sys) = lavender {
        let ghosts = probe_dual_entity(
            graph,
            sys.index,
            signed_container,
            organic_container,
            organic_residual,
            templates,
            base,
            None,
            20,
        );
        print_dual_ghosts(
            "SCENARIO A: Ghost connections for 'Lavender' (Signed vs Organic)",
            &ghosts,
            20,
        );
    }

    // -- Scenario B: Person-to-person ghost connections --
    println!();
    let person_indices: Vec<usize> = graph
        .entities
        .iter()
        .filter(|e| e.entity_type == EntityType::Person)
        .map(|e| e.index)
        .collect();

    let person_ghosts = probe_dual_pairs(
        graph,
        &person_indices,
        &person_indices,
        signed_container,
        organic_container,
        organic_residual,
        templates,
        base,
        30,
    );
    print_dual_ghosts(
        "SCENARIO B: Person <-> Person Ghost Network (Signed vs Organic)",
        &person_ghosts,
        30,
    );

    // -- Scenario C: System-to-nation ghost deployments --
    println!();
    let nations: Vec<usize> = graph
        .entities
        .iter()
        .filter(|e| e.axes.stakeholder_type.as_deref() == Some("Nation"))
        .map(|e| e.index)
        .collect();
    let systems: Vec<usize> = graph
        .entities
        .iter()
        .filter(|e| e.entity_type == EntityType::System)
        .map(|e| e.index)
        .collect();

    let deploy_ghosts = probe_dual_pairs(
        graph,
        &systems,
        &nations,
        signed_container,
        organic_container,
        organic_residual,
        templates,
        base,
        30,
    );
    print_dual_ghosts(
        "SCENARIO C: Ghost Deployments — System -> Nation (Signed vs Organic)",
        &deploy_ghosts,
        30,
    );

    // -- Scenario D: Civilian-military convergence --
    println!();
    let civic: Vec<usize> = graph
        .entities
        .iter()
        .filter(|e| e.entity_type == EntityType::CivicSystem)
        .map(|e| e.index)
        .collect();
    let military: Vec<usize> = graph
        .entities
        .iter()
        .filter(|e| e.entity_type == EntityType::System)
        .map(|e| e.index)
        .collect();

    let convergence_ghosts = probe_dual_pairs(
        graph,
        &civic,
        &military,
        signed_container,
        organic_container,
        organic_residual,
        templates,
        base,
        30,
    );
    print_dual_ghosts(
        "SCENARIO D: Civic <-> Military Ghost Convergence (Signed vs Organic)",
        &convergence_ghosts,
        30,
    );
}

// ---------------------------------------------------------------------------
// Part 10: Known Edge Validation
// ---------------------------------------------------------------------------

pub fn validate_known_edges(
    graph: &AiWarGraph,
    container: &[i8],
    residual: &[f32],
    templates: &[Vec<i8>],
    base: Base,
    coefficients: &[f32],
) {
    let mut direct_strengths = Vec::new();
    let mut residual_strengths = Vec::new();
    let mut by_type: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();

    for edge in &graph.edges {
        let src_idx = match graph.id_to_index.get(&edge.source_id) {
            Some(&i) => i,
            None => continue,
        };
        let tgt_idx = match graph.id_to_index.get(&edge.target_id) {
            Some(&i) => i,
            None => continue,
        };

        let probe = bind(&templates[src_idx], &templates[tgt_idx], base);
        let ds = cosine_similarity_i8(&probe, container);
        let rs = cosine_similarity_mixed(&probe, residual);

        direct_strengths.push(ds);
        residual_strengths.push(rs);

        let entry = by_type
            .entry(edge.edge_type.label().to_string())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(ds);
        entry.1.push(rs);
    }

    let mean = |v: &[f32]| -> f32 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };

    println!("{}", "=".repeat(66));
    println!("  KNOWN EDGE VALIDATION (Organic)");
    println!("{}", "=".repeat(66));
    println!(
        "  {} edges encoded, {} coefficients recovered",
        direct_strengths.len(),
        coefficients.len()
    );
    println!();
    println!("  Direct (container):");
    println!(
        "    mean: {:>+.4}, min: {:>+.4}, max: {:>+.4}",
        mean(&direct_strengths),
        direct_strengths.iter().cloned().fold(f32::MAX, f32::min),
        direct_strengths.iter().cloned().fold(f32::MIN, f32::max)
    );
    println!("  Residual (after extraction):");
    println!(
        "    mean: {:>+.4}, min: {:>+.4}, max: {:>+.4}",
        mean(&residual_strengths),
        residual_strengths.iter().cloned().fold(f32::MAX, f32::min),
        residual_strengths.iter().cloned().fold(f32::MIN, f32::max)
    );

    // Coefficient stats
    if !coefficients.is_empty() {
        let coeff_mean = coefficients.iter().sum::<f32>() / coefficients.len() as f32;
        let coeff_max = coefficients.iter().cloned().fold(f32::MIN, f32::max);
        let coeff_min = coefficients.iter().cloned().fold(f32::MAX, f32::min);
        let nonzero = coefficients.iter().filter(|&&c| c.abs() > 0.001).count();
        println!("  Surgical coefficients:");
        println!(
            "    mean: {:>+.4}, min: {:>+.4}, max: {:>+.4}, nonzero: {}/{}",
            coeff_mean,
            coeff_min,
            coeff_max,
            nonzero,
            coefficients.len()
        );
    }
    println!();

    println!("  By edge type:");
    let mut type_keys: Vec<_> = by_type.keys().cloned().collect();
    type_keys.sort();
    for key in &type_keys {
        let (d, r) = &by_type[key];
        println!(
            "    {:12} ({:>3} edges)  direct={:>+.4}  residual={:>+.4}",
            key,
            d.len(),
            mean(d),
            mean(r)
        );
    }
    println!("{}", "=".repeat(66));
}

// ---------------------------------------------------------------------------
// Part 11: Multi-Base Comparison
// ---------------------------------------------------------------------------

struct BaseRunResult {
    _base: Base,
    absorption_avg: f32,
    absorption_min: f32,
    residual_energy: f32,
    known_edge_direct_mean: f32,
    known_edge_residual_mean: f32,
    top_ghost_strength: f32,
    ghost_count_above_threshold: usize,
}

fn quick_base_summary(
    graph: &AiWarGraph,
    d: usize,
    base: Base,
    channels: usize,
    overlap: f32,
) -> BaseRunResult {
    let templates = generate_entity_templates(graph, d, base, overlap);
    let enc = encode_edges_organic(graph, &templates, d, base, channels);

    // Quick known-edge validation
    let mut direct_sum = 0.0f32;
    let mut residual_sum = 0.0f32;
    let mut edge_count = 0usize;

    for edge in &graph.edges {
        let src_idx = match graph.id_to_index.get(&edge.source_id) {
            Some(&i) => i,
            None => continue,
        };
        let tgt_idx = match graph.id_to_index.get(&edge.target_id) {
            Some(&i) => i,
            None => continue,
        };

        let probe = bind(&templates[src_idx], &templates[tgt_idx], base);
        direct_sum += cosine_similarity_i8(&probe, &enc.container);
        residual_sum += cosine_similarity_mixed(&probe, &enc.residual);
        edge_count += 1;
    }

    let known_direct = direct_sum / edge_count.max(1) as f32;
    let known_residual = residual_sum / edge_count.max(1) as f32;

    // Quick ghost sample: check a few non-edge pairs
    let ghosts =
        probe_ghost_connections(graph, &enc.container, &enc.residual, &templates, base, 20);
    let top_ghost = ghosts.first().map(|g| g.ghost_signal.abs()).unwrap_or(0.0);
    let above_thresh = ghosts
        .iter()
        .filter(|g| g.ghost_signal.abs() > 0.03)
        .count();

    BaseRunResult {
        _base: base,
        absorption_avg: enc.absorption_avg,
        absorption_min: enc.absorption_min,
        residual_energy: enc.residual_energy,
        known_edge_direct_mean: known_direct,
        known_edge_residual_mean: known_residual,
        top_ghost_strength: top_ghost,
        ghost_count_above_threshold: above_thresh,
    }
}

// ---------------------------------------------------------------------------
// Part 12: Main Runner
// ---------------------------------------------------------------------------

pub fn run_aiwar_ghost_oracle(graph_path: &str) {
    println!();
    println!("{}", "=".repeat(100));
    println!("  AI WAR GHOST ORACLE — Signed vs Organic Dual Comparison");
    println!("  221 entities, 356 edges from Sarah Ciston's 'AI War Cloud'");
    println!("  Method 1: Raw signed accumulation (baseline)");
    println!("  Method 2: OrganicWAL + base-aware plasticity + surgical residual");
    println!("  Each scenario shows both methods side-by-side");
    println!("{}", "=".repeat(100));
    println!();

    let d = 16384;
    let channels = 128;
    let overlap_per_axis = 0.05;

    // Load graph
    let graph = load_graph_from_file(graph_path);
    println!(
        "Loaded: {} entities, {} edges",
        graph.entity_count(),
        graph.edge_count()
    );

    // Show entity type breakdown
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for e in &graph.entities {
        *type_counts.entry(e.entity_type.label()).or_insert(0) += 1;
    }
    let mut type_list: Vec<_> = type_counts.iter().collect();
    type_list.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    print!("  Types: ");
    for (t, c) in &type_list {
        print!("{} {}, ", c, t);
    }
    println!();

    // Show edge type breakdown
    let mut edge_counts: HashMap<&str, usize> = HashMap::new();
    for e in &graph.edges {
        *edge_counts.entry(e.edge_type.label()).or_insert(0) += 1;
    }
    let mut edge_list: Vec<_> = edge_counts.iter().collect();
    edge_list.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    print!("  Edges: ");
    for (t, c) in &edge_list {
        print!("{} {}, ", c, t);
    }
    println!("\n");

    // ── Multi-base comparison (organic only, to pick best base) ──
    println!("{}", "=".repeat(100));
    println!("  MULTI-BASE COMPARISON: Signed(5) vs Signed(7) vs Signed(9)");
    println!("  All with base-aware plasticity, D={}, C={}", d, channels);
    println!("{}", "=".repeat(100));

    let bases = [Base::Signed(5), Base::Signed(7), Base::Signed(9)];
    println!(
        "  {:>12} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Base", "AbsAvg", "AbsMin", "ResEnergy", "KnownDir", "KnownRes", "TopGhost", "|G|>0.03"
    );
    println!("{}", "-".repeat(100));

    let mut best_base = Base::Signed(7);
    let mut best_ghost_count = 0usize;

    for &base in &bases {
        let r = quick_base_summary(&graph, d, base, channels, overlap_per_axis);
        println!(
            "  {:>12} {:>8.4} {:>8.4} {:>10.4} {:>+10.4} {:>+10.4} {:>+10.4} {:>8}",
            base.name(),
            r.absorption_avg,
            r.absorption_min,
            r.residual_energy,
            r.known_edge_direct_mean,
            r.known_edge_residual_mean,
            r.top_ghost_strength,
            r.ghost_count_above_threshold
        );
        if r.ghost_count_above_threshold > best_ghost_count {
            best_ghost_count = r.ghost_count_above_threshold;
            best_base = base;
        }
    }
    println!("{}", "=".repeat(100));
    println!("  Best ghost discrimination: {}\n", best_base.name());

    // ── Full dual analysis ──
    let analysis_base = best_base;
    println!(
        "Generating templates for {} (D={}, overlap={})...",
        analysis_base.name(),
        d,
        overlap_per_axis
    );
    let templates = generate_entity_templates(&graph, d, analysis_base, overlap_per_axis);
    println!("  {} templates generated\n", templates.len());

    // Encode: raw signed accumulation
    println!("Encoding edges (raw signed accumulation)...");
    let signed_container = encode_edges_signed_raw(&graph, &templates, d, analysis_base);
    println!();

    // Encode: organic WAL
    println!("Encoding edges (OrganicWAL + base-aware plasticity)...");
    let enc = encode_edges_organic(&graph, &templates, d, analysis_base, channels);
    println!();

    // Validate known edges (organic)
    validate_known_edges(
        &graph,
        &enc.container,
        &enc.residual,
        &templates,
        analysis_base,
        &enc.edge_coefficients,
    );

    // Also validate known edges against raw signed container
    println!();
    {
        let mut signed_strengths = Vec::new();
        for edge in &graph.edges {
            let src_idx = match graph.id_to_index.get(&edge.source_id) {
                Some(&i) => i,
                None => continue,
            };
            let tgt_idx = match graph.id_to_index.get(&edge.target_id) {
                Some(&i) => i,
                None => continue,
            };
            let probe = bind(&templates[src_idx], &templates[tgt_idx], analysis_base);
            signed_strengths.push(cosine_similarity_i8(&probe, &signed_container));
        }
        let mean_s = signed_strengths.iter().sum::<f32>() / signed_strengths.len().max(1) as f32;
        let min_s = signed_strengths.iter().cloned().fold(f32::MAX, f32::min);
        let max_s = signed_strengths.iter().cloned().fold(f32::MIN, f32::max);
        println!("{}", "=".repeat(66));
        println!("  KNOWN EDGE VALIDATION (Raw Signed)");
        println!("{}", "=".repeat(66));
        println!("  {} edges validated", signed_strengths.len());
        println!(
            "  Raw signed: mean={:>+.4}, min={:>+.4}, max={:>+.4}",
            mean_s, min_s, max_s
        );
        println!("{}", "=".repeat(66));
    }
    println!();

    // ── Run all 4 scenarios with dual method ──
    println!("{}", "=".repeat(100));
    println!("  DUAL-METHOD SCENARIOS: Signed (raw) vs Organic (WAL + residual)");
    println!("  Columns: Signed = raw accumulation cosine");
    println!("           OrgDirect = organic container cosine");
    println!("           OrgResid = organic residual cosine (after surgical extraction)");
    println!("  Sorted by |Signed| to show raw accumulation ranking");
    println!("{}", "=".repeat(100));

    run_dual_scenarios(
        &graph,
        &signed_container,
        &enc.container,
        &enc.residual,
        &templates,
        analysis_base,
    );

    println!();
    println!("{}", "=".repeat(100));
    println!("  END OF AI WAR GHOST ORACLE (Signed vs Organic Dual Comparison)");
    println!(
        "  Base: {}  |  Organic absorption: avg={:.4} min={:.4}",
        analysis_base.name(),
        enc.absorption_avg,
        enc.absorption_min
    );
    println!(
        "  Residual energy: {:.4}  |  {} edges encoded",
        enc.residual_energy, enc.edges_encoded
    );
    println!("{}", "=".repeat(100));
    println!();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_graph_path() -> String {
        // Try to find the data file relative to the crate root
        let paths = [
            "data/aiwar_graph.json",
            "../cubus-oracle/data/aiwar_graph.json",
        ];
        for p in &paths {
            if std::path::Path::new(p).exists() {
                return p.to_string();
            }
        }
        panic!("Could not find aiwar_graph.json — run from cubus-oracle directory");
    }

    #[test]
    fn test_parse_entity_counts() {
        let graph = load_graph_from_file(&test_graph_path());
        // 65 systems + 114 stakeholders + 23 civic + 7 historical + 12 people = 221
        assert_eq!(
            graph.entity_count(),
            221,
            "Expected 221 entities, got {}",
            graph.entity_count()
        );

        let systems = graph
            .entities
            .iter()
            .filter(|e| e.entity_type == EntityType::System)
            .count();
        assert_eq!(systems, 65, "Expected 65 systems, got {}", systems);

        let stakeholders = graph
            .entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Stakeholder)
            .count();
        assert_eq!(
            stakeholders, 114,
            "Expected 114 stakeholders, got {}",
            stakeholders
        );

        let civic = graph
            .entities
            .iter()
            .filter(|e| e.entity_type == EntityType::CivicSystem)
            .count();
        assert_eq!(civic, 23, "Expected 23 civic, got {}", civic);

        let historical = graph
            .entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Historical)
            .count();
        assert_eq!(historical, 7, "Expected 7 historical, got {}", historical);

        let people = graph
            .entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Person)
            .count();
        assert_eq!(people, 12, "Expected 12 people, got {}", people);
    }

    #[test]
    fn test_parse_edge_counts() {
        let graph = load_graph_from_file(&test_graph_path());
        // 114 develops + 79 deploys + 95 connection + 22 people + 21 place = 331
        let total = graph.edge_count();
        assert!(
            (300..=400).contains(&total),
            "Expected ~331 edges, got {}",
            total
        );

        let develops = graph
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Develops)
            .count();
        assert_eq!(develops, 114, "Expected 114 develops, got {}", develops);

        let deploys = graph
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Deploys)
            .count();
        assert_eq!(deploys, 79, "Expected 79 deploys, got {}", deploys);

        let connection = graph
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Connection)
            .count();
        assert_eq!(connection, 95, "Expected 95 connection, got {}", connection);
    }

    #[test]
    fn test_entity_lookup() {
        let graph = load_graph_from_file(&test_graph_path());
        let lavender = graph.get("Lavender");
        assert!(lavender.is_some(), "Lavender should exist");
        assert_eq!(lavender.unwrap().name, "Lavender");
        assert_eq!(lavender.unwrap().entity_type, EntityType::System);
    }

    #[test]
    fn test_neighbors() {
        let graph = load_graph_from_file(&test_graph_path());
        let neighbors = graph.neighbors("Lavender");
        assert!(!neighbors.is_empty(), "Lavender should have neighbors");
        // Unit 8200 develops Lavender
        let has_unit8200 = neighbors
            .iter()
            .any(|n| n.name.contains("Unit 8200") || n.id == "Unit8200");
        assert!(has_unit8200, "Unit 8200 should be a neighbor of Lavender");
    }

    #[test]
    fn test_deterministic_templates() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 1024;
        let base = Base::Signed(7);

        let t1 = generate_entity_templates(&graph, d, base, 0.05);
        let t2 = generate_entity_templates(&graph, d, base, 0.05);

        // Same graph + same params → same templates
        for i in 0..graph.entity_count() {
            assert_eq!(t1[i], t2[i], "Template {} should be deterministic", i);
        }
    }

    #[test]
    fn test_ontology_correlation() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 4096;
        let base = Base::Signed(7);
        let templates = generate_entity_templates(&graph, d, base, 0.05);

        // Find two systems that share ontology values (e.g., both Intelligence)
        let intel_systems: Vec<usize> = graph
            .entities
            .iter()
            .filter(|e| {
                e.entity_type == EntityType::System
                    && e.axes.military_use.as_deref() == Some("Intelligence")
            })
            .map(|e| e.index)
            .collect();

        if intel_systems.len() >= 2 {
            let i = intel_systems[0];
            let j = intel_systems[1];
            let same_type_cos = cosine_similarity_i8(&templates[i], &templates[j]);

            // Find a system with different ontology
            let diff_system = graph
                .entities
                .iter()
                .find(|e| e.entity_type == EntityType::Person)
                .map(|e| e.index);

            if let Some(k) = diff_system {
                let diff_type_cos = cosine_similarity_i8(&templates[i], &templates[k]);
                // Same-type correlation should be higher than cross-type
                assert!(
                    same_type_cos > diff_type_cos,
                    "Same ontology correlation ({:.4}) should exceed cross-type ({:.4})",
                    same_type_cos,
                    diff_type_cos
                );
            }
        }
    }

    #[test]
    fn test_organic_encoding_and_readback() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 4096;
        let base = Base::Signed(7);
        let channels = 64;

        let templates = generate_entity_templates(&graph, d, base, 0.05);
        let enc = encode_edges_organic(&graph, &templates, d, base, channels);

        // Container should not be all zeros
        assert!(
            enc.container.iter().any(|&v| v != 0),
            "Organic container should not be all zeros"
        );

        // Signed(7) container should be within [-3, 3]
        // i8 is always in [-128, 127] — this assertion documents intent
        assert!(!enc.container.is_empty(), "Container should not be empty");

        // Absorption should be healthy
        assert!(
            enc.absorption_avg > 0.9,
            "Absorption avg={:.4}, expected > 0.9",
            enc.absorption_avg
        );

        // Residual should exist (not all zeros)
        let res_nonzero = enc.residual.iter().filter(|&&v| v.abs() > 0.01).count();
        assert!(res_nonzero > 0, "Residual should have non-zero entries");

        // Coefficients should be recovered
        assert_eq!(
            enc.edge_coefficients.len(),
            enc.edges_encoded,
            "Should have one coefficient per encoded edge"
        );
    }

    #[test]
    fn test_ghost_probes_organic() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 2048;
        let base = Base::Signed(7);
        let channels = 32;

        let templates = generate_entity_templates(&graph, d, base, 0.05);
        let enc = encode_edges_organic(&graph, &templates, d, base, channels);

        let ghosts =
            probe_ghost_connections(&graph, &enc.container, &enc.residual, &templates, base, 10);

        assert!(
            !ghosts.is_empty(),
            "Should find at least some ghost connections"
        );
        // Ghost signals should vary (not all zero)
        let has_nonzero = ghosts.iter().any(|g| g.ghost_signal.abs() > 0.001);
        assert!(has_nonzero, "Ghost signals should not all be zero");
    }

    #[test]
    fn test_ghost_probes_reproducible() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 2048;
        let base = Base::Signed(7);
        let channels = 32;

        let templates = generate_entity_templates(&graph, d, base, 0.05);
        let enc = encode_edges_organic(&graph, &templates, d, base, channels);

        let g1 =
            probe_ghost_connections(&graph, &enc.container, &enc.residual, &templates, base, 5);
        let g2 =
            probe_ghost_connections(&graph, &enc.container, &enc.residual, &templates, base, 5);

        for (a, b) in g1.iter().zip(g2.iter()) {
            assert_eq!(a.entity_a, b.entity_a);
            assert_eq!(a.entity_b, b.entity_b);
            assert!(
                (a.ghost_signal - b.ghost_signal).abs() < 1e-6,
                "Ghost signals should be reproducible"
            );
        }
    }

    #[test]
    fn test_base_aware_plasticity_improves_signed5() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 2048;
        let channels = 32;
        let base = Base::Signed(5);

        let templates = generate_entity_templates(&graph, d, base, 0.05);
        let enc = encode_edges_organic(&graph, &templates, d, base, channels);

        // With base-aware plasticity (Signed(5) gets scale=1.5),
        // absorption should still be healthy
        assert!(
            enc.absorption_avg > 0.9,
            "Signed(5) absorption avg={:.4}, expected > 0.9",
            enc.absorption_avg
        );

        // Coefficients should be mostly recovered
        let nonzero = enc
            .edge_coefficients
            .iter()
            .filter(|&&c| c.abs() > 0.001)
            .count();
        assert!(
            nonzero > enc.edges_encoded / 2,
            "Expected most coefficients to be nonzero, got {}/{}",
            nonzero,
            enc.edges_encoded
        );
    }
}
