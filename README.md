# lau-information-geometry-agents

**Information geometry for agent belief spaces: Fisher metric, geodesics, natural gradient, Amari's dual connections, and Chentsov's uniqueness theorem.**

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-118-green.svg)]()

## What This Does

This crate treats agent belief spaces as **statistical manifolds** — Riemannian manifolds where the metric is the Fisher information matrix. Learning becomes geodesic motion on this manifold, and the curvature of belief space determines how hard learning is.

The crate provides:

- **Fisher information metric** — the natural distance between probability distributions
- **Fisher-Rao geodesic distance** — the "true" distance between beliefs (not KL divergence)
- **Natural gradient descent** — steepest descent that respects the geometry of belief space
- **Amari's α-connections** — a family of connections that includes the exponential (e-) and mixture (m-) connections
- **Chentsov's theorem** — the Fisher metric is the *unique* invariant metric on the probability simplex
- **Curvature-aware learning rates** — positive curvature = easy learning, negative = hard

**118 tests** cover every module from manifold construction through curvature-adaptive learning.

## Key Idea

Standard gradient descent treats all parameter directions equally. But on the probability simplex, moving 0.01 in probability mass from a low-probability event to a high-probability event is *not* the same as the reverse. The **Fisher information metric** captures this asymmetry:

```
g_ij(θ) = E[∂ᵢ log p(x;θ) · ∂ⱼ log p(x;θ)]
```

The **natural gradient** `G⁻¹∇L` respects this geometry, giving the true steepest descent direction. And by **Chentsov's theorem**, the Fisher metric is the *only* metric on the probability simplex that is invariant under sufficient statistics (Markov morphisms). There is no other consistent choice.

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-information-geometry-agents = { git = "https://github.com/SuperInstance/lau-information-geometry-agents" }
```

### Dependencies

- `nalgebra` — linear algebra (matrices, Cholesky, eigenvalues)
- `serde` / `serde_json` — serialization
- `rand` — random number generation

## Quick Start

```rust
use lau_information_geometry_agents::*;

// Create a Normal distribution manifold N(μ, σ²) parameterized by (μ, log σ)
let manifold = NormalManifold::new();
let theta = NormalManifold::params(0.0, 1.0);  // N(0, 1)

// Fisher information matrix at this point
let fisher = manifold.fisher_information(&theta);
// g_μμ = 1, g_{logσ, logσ} = 2, off-diagonal = 0

// Fisher-Rao distance between two Normal distributions
let d = normal_fisher_rao(0.0, 1.0, 1.0, 1.0);
// Distance between N(0,1) and N(1,1)

// Natural gradient descent step
let theta = NormalManifold::params(2.0, 0.5);  // current belief
let grad = vec![1.0, -0.5];                      // Euclidean gradient
let new_theta = natural_gradient_step(&manifold, &theta, &grad, 0.1);

// Check curvature of belief space
let K = scalar_curvature(&manifold, &theta, 1e-5);

// Jeffreys prior (uniform in Fisher metric): π(θ) ∝ √det(g)
let log_prior = JeffreysPrior::log_unnormalized(&manifold, &theta);

// Chentsov's theorem: verify Fisher metric is the unique invariant metric
let candidate = 2.0 * fisher.clone();  // scaled Fisher
assert!(verify_chentsov(&fisher, &candidate, 1e-10));  // proportional → passes

// Create an agent and train with natural gradient
let mut agent = BeliefAgent::new(theta, 0.1).with_adaptive_lr();
agent.natural_gradient_update(&manifold, &grad);
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `ManifoldPoint` | A point θ ∈ ℝⁿ on a statistical manifold (parameter vector) |
| `StatisticalManifold` | Trait: defines pdf, log_pdf, score, Fisher info, valid params |

### `manifold` — Concrete Statistical Manifolds

| Manifold | Parameterization | Dimension |
|----------|-----------------|-----------|
| `NormalManifold` | θ = (μ, log σ) | 2 |
| More can be added by implementing `StatisticalManifold` | | |

```rust
pub trait StatisticalManifold: Send + Sync {
    fn dimension(&self) -> usize;
    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64;
    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64;
    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64>;  // ∂ log p / ∂θ
    fn fisher_information(&self, theta: &ManifoldPoint) -> DMatrix<f64>;
    fn valid_params(&self, theta: &ManifoldPoint) -> bool;
}
```

### `fisher_metric` — Fisher Information Metric

| Function | Description |
|----------|-------------|
| `fisher_metric_numerical(manifold, θ, x_min, x_max, n)` | Numerical integration of score outer product |
| `is_positive_definite(g)` | Check via Cholesky decomposition |
| `volume_element(g)` | √det(g) — Riemannian volume density |
| `fisher_trace(g)` | Total Fisher information content |
| `fisher_condition_number(g)` | Condition number of the metric tensor |

### `fisher_rao` — Fisher-Rao Geodesic Distance

The Fisher-Rao distance is the geodesic distance under the Fisher metric:

```
d_FR(θ₁, θ₂) = inf ∫ √(θ̇ᵀ g(θ) θ̇) dt
```

| Function | Description |
|----------|-------------|
| `normal_fisher_rao(μ₁, σ₁, μ₂, σ₂)` | **Closed form** for Normal distributions: `√2 · arccos(√(2σ₁σ₂/(σ₁²+σ₂²)) · exp(-(Δμ)²/(4(σ₁²+σ₂²))))` |
| `fisher_rao_numerical(manifold, θ₁, θ₂, n_segments)` | Numerical approximation for arbitrary manifolds |

### `natural_gradient` — Natural Gradient Descent

The natural gradient is the steepest descent direction in the Fisher metric:

```
∇̃L = G⁻¹ ∇L
```

| Function | Description |
|----------|-------------|
| `natural_gradient(fisher, euclidean_grad)` | Compute G⁻¹∇L (with regularization fallback) |
| `natural_gradient_step(manifold, θ, grad, lr)` | Single descent step: `θ' = θ - η·G⁻¹∇L` |
| `riemannian_gradient_norm(fisher, grad)` | √(gᵀ G g) — true gradient magnitude |

### `alpha_connections` — Amari's α-Connections

A one-parameter family of affine connections on the statistical manifold:

```
Γ^α_{ij,k} = E[(∂ᵢ∂ⱼ l + (1-α)/2 · ∂ᵢl · ∂ⱼl) · ∂ₖl]
```

| α | Connection | Name |
|---|-----------|------|
| +1 | Exponential (e-) | Dually flat for exponential families |
| 0 | Levi-Civita | Riemannian (metric) connection |
| -1 | Mixture (m-) | Dually flat for mixture families |

| Function | Description |
|----------|-------------|
| `alpha_christoffel_first_kind(manifold, θ, α, ...)` | Christoffel symbols Γ_{ij,k} for given α |
| `alpha_christoffel_second_kind(manifold, θ, α, ...)` | Raised-index symbols Γ^k_{ij} |

### `dual_connections` — e-connection / m-connection Duality

The e-connection and m-connection are **dual** with respect to the Fisher metric:

```
∂_k g_{ij} = Γ^{(e)}_{ij,k} + Γ^{(m)}_{ij,k}
```

| Type/Method | Description |
|-------------|-------------|
| `DualConnections::compute(manifold, θ, ...)` | Compute both e- and m-connection Christoffel symbols |
| `.verify_duality(manifold, θ, eps)` | Numerically verify the duality relation |
| `.e_christoffel` / `.m_christoffel` | The two connection tensors |

### `curvature` — Riemann Curvature of Belief Space

| Function | Description |
|----------|-------------|
| `riemann_curvature(manifold, θ, eps)` | Riemann curvature tensor (numerical) |
| `ricci_curvature(manifold, θ, eps)` | Ricci curvature (trace of Riemann) |
| `scalar_curvature(manifold, θ, eps)` | Scalar curvature (trace of Ricci) |
| `sectional_curvature(manifold, θ, v, w, eps)` | Sectional curvature in a 2-plane |

**Interpretation**:
- Positive curvature → beliefs converge easily (the space "closes in")
- Negative curvature → beliefs diverge (the space "opens up")
- Zero curvature → flat (exponential families in natural parameters)

### `jeffreys_prior` — Jeffreys Prior

The unique reparameterization-invariant prior:

```
π(θ) ∝ √det(g(θ))
```

| Method | Description |
|--------|-------------|
| `JeffreysPrior::unnormalized(manifold, θ)` | √det(g(θ)) |
| `JeffreysPrior::log_unnormalized(manifold, θ)` | ½ log det(g(θ)) |
| `JeffreysPrior::normalize_1d(...)` / `.normalize_2d(...)` | Estimate normalization constant |

### `chentsov` — Chentsov's Uniqueness Theorem

The Fisher information metric is the **unique** Riemannian metric on the statistical manifold that is invariant under all Markov morphisms (stochastic maps / sufficient statistics).

| Function | Description |
|----------|-------------|
| `verify_chentsov(fisher, candidate, tol)` | Check if a candidate metric is proportional to Fisher |
| `MarkovMorphism` | A row-stochastic matrix representing a coarse-graining map |
| `MarkovMorphism::merge_categories(n, i, j)` | Merge two categories into one |
| `pullback_metric(morphism, metric)` | Pull back a metric through a Markov morphism |
| `verify_invariance(fisher, morphism, tol)` | Check invariance under a specific morphism |

### `monotonicity` — Information Monotonicity

Coarse-graining (data processing) can only reduce Fisher information:

```
g_coarse ≤ g_full  (in Loewner order)
```

| Function | Description |
|----------|-------------|
| `verify_monotonicity(g_full, g_coarse)` | Check the monotonicity inequality |
| `coarse_grain_categorical(n, groups, probs)` | Aggregate categories |
| `fisher_coarse_grained(n, groups, probs)` | Fisher info after coarse-graining |

### `agent_learning` — Agents with Geometric Learning

| Type/Method | Description |
|-------------|-------------|
| `BeliefAgent::new(belief, lr)` | Create agent at a belief state with learning rate |
| `.with_adaptive_lr()` | Enable curvature-adaptive learning rates |
| `.natural_gradient_update(manifold, grad)` | Single natural gradient step |
| `.trajectory` | History of belief states |
| `.distance_traveled` | Cumulative Fisher-Rao distance |

Curvature-adaptive learning: increase step size where curvature is positive (easy learning), decrease where negative (hard learning).

## How It Works

### 1. Parameterize the Belief Space

Agent beliefs are probability distributions `p(x; θ)` parameterized by θ ∈ ℝⁿ. Different parameterizations give different geometries — the Fisher metric ensures the geometry is **intrinsic** (independent of parameterization, by Chentsov's theorem).

### 2. Compute the Fisher Metric

```
g_ij(θ) = E[∂ᵢ log p(x;θ) · ∂ⱼ log p(x;θ)]
```

For Normal(μ, σ²) with θ = (μ, log σ):
```
g = [1/σ²  0]
    [0      2]
```

### 3. Measure Belief Distances with Fisher-Rao

The geodesic distance between two beliefs is:
```
d_FR(θ₁, θ₂) = √2 · arccos(√(2σ₁σ₂/(σ₁²+σ₂²)) · exp(-(Δμ)²/(4(σ₁²+σ₂²))))
```

This is **not** KL divergence (which is not symmetric and not a true metric).

### 4. Optimize with Natural Gradient

Standard gradient descent treats all directions equally. Natural gradient:
```
θ_{t+1} = θ_t - η · G⁻¹(θ_t) · ∇L(θ_t)
```
respects the geometry, giving faster and more stable convergence on curved manifolds.

### 5. Analyze Curvature

Scalar curvature of the belief manifold:
- K > 0: "sphere-like" — beliefs converge easily
- K < 0: "saddle-like" — beliefs diverge, learning is hard
- K = 0: flat — exponential families in natural parameters

### 6. Use Jeffreys Prior for Bayesian Agents

The Jeffreys prior `π(θ) ∝ √det(g(θ))` is the unique prior that doesn't depend on how you parameterize the beliefs. It assigns more mass to regions with more "information resolution."

## The Math

### The Fisher Information Metric

For a parametric family {p(x;θ) : θ ∈ Θ ⊂ ℝⁿ}, the Fisher information matrix is:

```
g_ij(θ) = ∫ ∂ᵢ log p(x;θ) · ∂ⱼ log p(x;θ) · p(x;θ) dx
         = -E[∂ᵢ∂ⱼ log p(x;θ)]
```

Properties:
- Symmetric positive definite
- Invariant under reparameterization (pullback)
- Determines the Cramér-Rao bound: Var(θ̂) ≥ G⁻¹

### Chentsov's Theorem (1972)

**Theorem**: The Fisher information metric is the unique Riemannian metric on the simplex Δₙ₋₁ = {p ∈ ℝⁿ : pᵢ ≥ 0, Σpᵢ = 1} that is invariant under all Markov morphisms (stochastic maps that coarse-grain the categories).

This is a profound uniqueness result: there is literally no other consistent choice of metric for probability distributions.

### Amari's α-Connections

The α-connection interpolates between three fundamental connections:

```
Γ^α_{ij,k} = E[(∂ᵢ∂ⱼl + (1-α)/2 · ∂ᵢl · ∂ⱼl) · ∂ₖl]
```

- α = +1 (e-connection): flat for exponential families; natural parameters are affine coordinates
- α = -1 (m-connection): flat for mixture families; mixture parameters are affine coordinates
- α = 0 (Levi-Civita): the unique metric-compatible, torsion-free connection

The e-connection and m-connection are **dual**: `∂_k g_{ij} = Γ^{(e)}_{ij,k} + Γ^{(m)}_{ij,k}`.

### Natural Gradient (Amari, 1998)

The steepest descent direction on a Riemannian manifold (M, g) is:

```
∇̃f = G⁻¹∇f
```

This follows from minimizing `df(v) = ∇f · v` subject to `||v||_g = 1`, i.e., `vᵀ G v = 1`.

### Jeffreys Prior

```
π(θ) ∝ √det(g(θ))
```

This is the volume density of the Fisher metric. It is:
- Reparameterization-invariant
- Uniform in the intrinsic geometry
- The "uninformative" prior that doesn't favor any parameterization

### Information Monotonicity

If T is a coarse-graining map (Markov kernel), then:

```
Fisher(Tθ) ≤ Fisher(θ)  (in Loewner order)
```

You cannot create information by processing data. This is the geometric version of the data processing inequality.

## Project Structure

```
src/
├── lib.rs              # Crate root: ManifoldPoint, StatisticalManifold trait, re-exports
├── manifold.rs         # Concrete manifolds: NormalManifold
├── fisher_metric.rs    # Fisher information matrix computation and properties
├── fisher_rao.rs       # Fisher-Rao geodesic distance (closed form + numerical)
├── natural_gradient.rs # Natural gradient descent: G⁻¹∇L
├── alpha_connections.rs # Amari's α-connections (Christoffel symbols)
├── dual_connections.rs # e-/m-connection duality
├── curvature.rs        # Riemann, Ricci, scalar curvature
├── jeffreys_prior.rs   # Jeffreys prior: √det(g)
├── chentsov.rs         # Chentsov's uniqueness theorem verification
├── monotonicity.rs     # Information monotonicity under coarse-graining
└── agent_learning.rs   # BeliefAgent with natural gradient + adaptive learning rates
```

## License

MIT
