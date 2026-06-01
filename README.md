# lau-information-geometry-agents

> Geometry of belief: Fisher metrics, Amari connections, and natural-gradient learning on statistical manifolds.

## What This Does

This crate treats agent belief states as points on **statistical manifolds** — smooth geometric spaces where each point is a probability distribution. It computes the Fisher information metric, geodesic distances, natural gradients, and Amari's α-connections so that learning algorithms respect the intrinsic geometry of probability distributions rather than imposing arbitrary Euclidean structure.

If you're building agents that learn by updating probability distributions, this gives you the differential-geometric toolkit to do it correctly: steepest descent becomes natural gradient, straight lines become geodesics, and belief updates follow the curvature of the space.

## The Key Idea

Probability distributions aren't vectors in Euclidean space — they live on a curved manifold. The **Fisher information matrix** is the natural inner product on this manifold, and it makes some directions "longer" than others (directions where small parameter changes cause big distributional shifts). **Natural gradient** adjusts for this: it points in the steepest direction *as measured by the Fisher metric*, not the Euclidean one. This is why natural gradient descent converges faster and more reliably than vanilla gradient descent for probabilistic models.

Think of it like navigating on Earth: the shortest path between two cities isn't a straight line on a flat map, it's a great circle on the globe. The Fisher metric is the "globe" for probability distributions.

## Install

```bash
cargo add lau-information-geometry-agents
```

## Quick Start

```rust
use lau_information_geometry_agents::*;

fn main() {
    // Create a Normal(μ, σ²) manifold
    let manifold = manifold::NormalManifold;

    // Two belief states: N(0,1) and N(1,1)
    let belief_a = NormalManifold::params(0.0, 1.0);
    let belief_b = NormalManifold::params(1.0, 1.0);

    // Fisher information at a belief point
    let fisher = manifold.fisher_information(&belief_a);
    println!("Fisher info (μ, logσ):\n{}", fisher);
    // [[1.0, 0.0], [0.0, 2.0]]  (for σ=1)

    // Fisher-Rao geodesic distance between beliefs
    let dist = fisher_rao::normal_fisher_rao(0.0, 1.0, 1.0, 1.0);
    println!("Fisher-Rao distance: {:.4}", dist);

    // Natural gradient descent step
    let euclidean_grad = vec![2.0, -0.5];
    let new_belief = natural_gradient::natural_gradient_step(
        &manifold, &belief_a, &euclidean_grad, 0.1
    );
    println!("Updated belief: μ={:.3}, σ={:.3}",
        new_belief.theta[0], new_belief.theta[1].exp());

    // Agent with curvature-adaptive learning
    let mut agent = agent_learning::BeliefAgent::new(belief_a, 0.01)
        .with_adaptive_lr();
    agent.natural_gradient_update(&manifold, &euclidean_grad);
    println!("Steps taken: {}", agent.steps());
}
```

## API Reference

### Core Types

#### `ManifoldPoint`
A point on a statistical manifold, parameterized by θ ∈ ℝⁿ.

```rust
let p = ManifoldPoint::new(vec![0.0, 1.0]); // (μ=0, logσ=1)
let dim = p.dim(); // 2
let v = p.to_dvector(); // nalgebra DVector
```

#### `StatisticalManifold` (trait)
The fundamental trait for parametric distribution families. Every manifold implements:

| Method | Returns | Description |
|--------|---------|-------------|
| `dimension()` | `usize` | Parameter space dimension |
| `pdf(x, θ)` | `f64` | Probability density p(x; θ) |
| `log_pdf(x, θ)` | `f64` | Log-density log p(x; θ) |
| `score(x, θ)` | `Vec<f64>` | Score function ∂/∂θ log p(x; θ) |
| `fisher_information(θ)` | `DMatrix<f64>` | Fisher information matrix G(θ) |
| `valid_params(θ)` | `bool` | Whether parameters are in the valid set |

### Manifold Implementations

#### `NormalManifold`
Univariate Normal N(μ, σ²), parameterized as θ = (μ, log σ).

```rust
let m = NormalManifold;
let theta = NormalManifold::params(0.0, 1.0); // N(0,1)
let mu = NormalManifold::mu(&theta);   // 0.0
let sigma = NormalManifold::sigma(&theta); // 1.0
```

Fisher information: g = diag(1/σ², 2) — diagonal, so μ and σ learning decouple.

#### `CategoricalManifold`
Categorical distribution on {1,...,K} via softmax parameterization.

```rust
let m = CategoricalManifold::new(4); // 4 categories, 3 free params
let theta = ManifoldPoint::new(vec![0.5, -0.3, 1.0]);
let probs = m.to_probs(&theta); // softmax probabilities
```

#### `ExponentialManifold`
Exponential distribution with rate λ, parameterized as θ = log λ.

```rust
let m = ExponentialManifold;
let theta = ExponentialManifold::params(2.0); // rate = 2
```

Fisher information: g = 1 (flat manifold — geodesics are straight lines in log λ space).

#### `PoissonManifold`
Poisson distribution with rate λ, θ = log λ. Fisher info: g = 1 (flat).

#### `BetaManifold`
Beta distribution B(α, β), θ = (log α, log β). Uses Lanczos approximation for log-gamma and asymptotic expansions for digamma/trigamma.

```rust
let m = BetaManifold;
let theta = BetaManifold::params(2.0, 3.0);
```

#### `GammaManifold`
Gamma distribution Γ(α, β) with shape α and rate β, θ = (log α, log β).

#### `DiagonalNormalManifold`
Multivariate Normal with diagonal covariance. θ = (μ₁,...,μₖ, log σ₁,..., log σₖ).

### Fisher Metric (`fisher_metric`)

#### `fisher_metric_numerical`
Compute Fisher information via numerical integration of the score function.

```rust
let g = fisher_metric_numerical(&manifold, &theta, -10.0, 10.0, 10000);
```

#### `is_positive_definite`
Check via Cholesky decomposition.

```rust
assert!(is_positive_definite(&fisher));
```

#### `volume_element`
Riemannian volume density √det(g).

#### `fisher_trace`
Total Fisher information content (trace of G).

#### `fisher_condition_number`
Ratio of largest to smallest eigenvalue — measures parameterization quality.

### Fisher-Rao Distance (`fisher_rao`)

#### `normal_fisher_rao`
Closed-form geodesic distance between two Normal distributions:

```
d = √2 · arccos(√(2σ₁σ₂/(σ₁²+σ₂²)) · exp(-(μ₁-μ₂)²/(4(σ₁²+σ₂²))))
```

```rust
let d = normal_fisher_rao(0.0, 1.0, 1.0, 2.0);
```

#### `fisher_rao_numerical`
Numerical geodesic distance via path energy along a piecewise-linear path.

```rust
let d = fisher_rao_numerical(&manifold, &theta1, &theta2, 100);
```

#### `bhattacharyya_distance_normal`
Bhattacharyya distance between Normals (fast approximation).

#### `hellinger_distance_normal`
Hellinger distance H(p,q) = √(1 - BC(p,q)).

### Natural Gradient (`natural_gradient`)

#### `natural_gradient`
Compute Ñf = G⁻¹∇f, with regularization for near-singular Fisher matrices.

```rust
let ng = natural_gradient(&fisher, &euclidean_grad);
```

#### `natural_gradient_step`
One step of natural gradient descent: θ ← θ - η · G⁻¹∇L.

```rust
let new_theta = natural_gradient_step(&manifold, &theta, &grad, 0.01);
```

#### `riemannian_gradient_norm`
Riemannian norm √(gᵀGg) of a Euclidean gradient.

#### `kl_gradient_approx`
Approximate KL divergence gradient: G·(θ - θ').

### Amari α-Connections (`alpha_connections`)

#### `alpha_christoffel_first_kind`
Christoffel symbols Γ^α_{ij,k} for Amari's α-connection:

- α = +1: exponential (e-) connection
- α = -1: mixture (m-) connection  
- α = 0: Levi-Civita (metric) connection

```rust
let gamma = alpha_christoffel_first_kind(&manifold, &theta, 1.0, -10.0, 10.0, 1000);
```

#### `alpha_christoffel_second_kind`
Raised-index Christoffel symbols: Γ^α_{ij}^k = g^{kl} Γ_{ij,l}.

#### `alpha_geodesic`
Geodesic interpolation between two parameter points under the α-connection.

```rust
let midpoint = alpha_geodesic(&theta1, &theta2, 1.0, 0.5);
```

#### `alpha_divergence`
Amari's α-divergence: D_α(p||q) = 4/(1-α²) · [1 - ∫ p^{(1+α)/2} q^{(1-α)/2} dx].

#### `parallel_transport_alpha`
Parallel transport a vector along a curve using α-connection coefficients.

### Dual Connections (`dual_connections`)

#### `DualConnections`
The e-connection (α=+1) and m-connection (α=-1) as a dual pair, with duality verification.

```rust
let dc = DualConnections::compute(&manifold, &theta, -10.0, 10.0, 1000);
let max_violation = dc.verify_duality(&manifold, &theta, 1e-4);
```

#### `e_geodesic` / `m_geodesic`
Straight-line interpolation in natural (η) and expectation (μ) parameter spaces.

#### `e_projection` / `m_projection`
Find the nearest submanifold point minimizing KL(p||q) and KL(q||p) respectively.

#### `generalized_pythagorean`
Verify the information-geometric Pythagorean theorem: if an e-geodesic and m-geodesic meet orthogonally, then D(p||r) = D(p||q) + D(q||r).

### Curvature (`curvature`)

#### `scalar_curvature`
Scalar curvature of the statistical manifold. Positive curvature → beliefs converge (easy learning). Negative → beliefs diverge (hard learning).

```rust
let k = scalar_curvature(&manifold, &theta);
```

#### `riemann_curvature`
Riemann curvature tensor components computed via numerical differentiation of the metric.

#### `ricci_curvature_direction`
Ricci curvature in a given tangent direction.

#### `learning_difficulty`
Negative of scalar curvature — a direct measure of how hard learning is in this region of belief space.

#### `sectional_curvature`
Sectional curvature between two tangent vectors (measures curvature of the 2-plane they span).

### Jeffreys Prior (`jeffreys_prior`)

#### `JeffreysPrior`
The reparameterization-invariant prior π(θ) ∝ √det(g(θ)).

```rust
let prior = JeffreysPrior::normalize_1d(&manifold, -5.0, 5.0, 1000);
let density = prior.density(&manifold, &theta);
```

#### `verify_reparameterization_invariance`
Check that the Jeffreys prior transforms correctly under coordinate changes.

### Chentsov's Theorem (`chentsov`)

#### `verify_chentsov`
Verify that a candidate metric on the probability simplex is proportional to the Fisher information metric (the unique invariant metric by Chentsov's theorem).

#### `MarkovMorphism`
A stochastic map T: Δ_{n-1} → Δ_{m-1}, including identity, permutation, and coarse-graining morphisms.

```rust
let morphism = MarkovMorphism::merge_categories(4, 0, 1);
let new_probs = morphism.apply(&probs);
```

#### `InformationGeometryAxioms`
Verify symmetry, positive definiteness, and Chentsov uniqueness of candidate metrics.

### Information Monotonicity (`monotonicity`)

#### `verify_monotonicity`
Verify that coarse-graining reduces Fisher information (data processing inequality).

#### `coarse_grain_categorical`
Aggregate categories in a categorical distribution.

#### `data_processing_inequality`
Check whether a statistic is sufficient (preserves Fisher info) or lossy.

#### `cramer_rao_bound`
Compute the Cramér-Rao lower bound I(θ)⁻¹ on estimator variance.

#### `verify_cramer_rao`
Check if an estimator achieves the Cramér-Rao bound (efficiency).

### Agent Learning (`agent_learning`)

#### `BeliefAgent`
An agent with beliefs on a statistical manifold, using natural gradient updates.

```rust
let mut agent = BeliefAgent::new(theta, 0.01).with_adaptive_lr();
agent.natural_gradient_update(&manifold, &grad);
println!("Distance traveled: {}", agent.distance_traveled);
```

#### `GeometricLearningSchedule`
Curvature-aware learning rate schedule: lr = base_lr / (1 + curvature_scale · |κ|).

```rust
let schedule = GeometricLearningSchedule::conservative();
let lr = schedule.learning_rate(curvature);
```

#### `MultiAgentBeliefSystem`
Multi-agent consensus via Fisher-weighted barycenter, with disagreement measurement.

```rust
let mut system = MultiAgentBeliefSystem::new(vec![agent1, agent2, agent3]);
let consensus = system.consensus(&manifold);
let disagreement = system.disagreement(&manifold);
system.belief_averaging_step(&manifold, 0.1);
```

## How It Works

**Fisher information** is computed analytically for each distribution family (Normal, Categorical, Exponential, etc.) and numerically via score-function integration as a cross-check. The Fisher matrix G(θ) measures how sensitive the distribution is to parameter changes — directions where the distribution changes a lot have high Fisher information.

**Natural gradient** multiplies the Euclidean gradient by G⁻¹, which corrects for the non-uniform "stretching" of parameter space. When G has large eigenvalues (high information), the natural gradient takes smaller steps in that direction — you don't need big steps where the distribution is already sensitive.

**α-connections** are computed numerically using finite-difference second derivatives of the score function. The Christoffel symbols Γ^α_{ij,k} depend on the choice of α through the coupling term (1-α)/2 · ∂ᵢl · ∂ⱼl, which interpolates between the e-connection and m-connection.

**Geodesic distances** use the closed-form formula for Normal distributions (via the spherical embedding trick). For general manifolds, the distance is approximated by discretizing a path and summing √(vᵀGv)·dt.

**Curvature** is computed via numerical second derivatives of the Fisher metric tensor, following the standard Riemannian curvature formulas. For 2D manifolds, the Gaussian curvature is extracted; for higher dimensions, scalar curvature is the trace of the Ricci tensor.

## The Math

### Fisher Information Matrix

For a parametric family p(x; θ):

$$g_{ij}(\theta) = \mathbb{E}_{X \sim p(\cdot;\theta)}\left[\frac{\partial}{\partial\theta_i} \log p(X;\theta) \cdot \frac{\partial}{\partial\theta_j} \log p(X;\theta)\right]$$

### Natural Gradient

$$\tilde{\nabla} f = G(\theta)^{-1} \nabla f(\theta)$$

This is the steepest ascent direction under the constraint $\|d\theta\|_G = \sqrt{d\theta^T G \, d\theta} \leq \epsilon$.

### Amari α-Connection

$$\Gamma^{(\alpha)}_{ij,k} = \mathbb{E}\left[\left(\partial_i \partial_j \ell + \frac{1-\alpha}{2} \partial_i \ell \cdot \partial_j \ell\right) \partial_k \ell\right]$$

where $\ell = \log p(x;\theta)$.

### Fisher-Rao Distance (Normal)

$$d_{FR}\bigl(N(\mu_1,\sigma_1^2),\, N(\mu_2,\sigma_2^2)\bigr) = \sqrt{2} \arccos\left(\sqrt{\frac{2\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2}} \exp\left(-\frac{(\mu_1-\mu_2)^2}{4(\sigma_1^2+\sigma_2^2)}\right)\right)$$

### Chentsov's Theorem

The Fisher information metric is (up to a positive constant) the **unique** Riemannian metric on the probability simplex that is invariant under all Markov morphisms (stochastic maps / sufficient statistics).

### Cramér-Rao Bound

For any unbiased estimator $\hat{\theta}$:

$$\text{Var}(\hat{\theta}) \geq G(\theta)^{-1}$$

An estimator is **efficient** if it achieves equality (e.g., MLE asymptotically).

### Jeffreys Prior

$$\pi(\theta) \propto \sqrt{\det G(\theta)}$$

This is the unique prior invariant under smooth reparameterization: if $\phi = f(\theta)$, then $\pi(\phi) = \pi(\theta) \cdot |d\theta/d\phi|$.

## License

MIT
