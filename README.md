# lau-information-geometry-agents

> Fisher metric, Amari α-connections, and natural gradient descent for agent belief updating

## What This Does

This crate implements information geometry — the mathematics of treating probability distributions as points on a Riemannian manifold — applied to agent learning. The Fisher information matrix serves as the metric tensor, Amari's α-connections generalize the affine structure, and natural gradient descent uses the Fisher metric to find steeper descent directions than Euclidean gradient.

## The Key Idea

Standard gradient descent treats parameter space as flat. But parameter space isn't flat — small changes to probabilities near 0 or 1 have vastly different effects. The Fisher information matrix measures this curvature. Natural gradient (F⁻¹∇) accounts for curvature, just like Newton's method accounts for Hessian curvature — but Fisher is always positive semi-definite, so you never go uphill by accident. The `BeliefAgent` uses this to update its beliefs optimally.

## Install

```toml
[dependencies]
lau-information-geometry-agents = { git = "https://github.com/SuperInstance/lau-information-geometry-agents" }
```

## Quick Start

```rust
use lau_information_geometry_agents::*;
use nalgebra::DVector;

// Create a belief agent with 4 states
let mut agent = BeliefAgent::new(4);
agent.set_belief(DVector::from_vec(vec![0.1, 0.4, 0.3, 0.2]));

// Observe evidence (likelihood for each state)
let likelihood = DVector::from_vec(vec![0.01, 0.8, 0.1, 0.09]);

// Natural gradient update (respects the geometry of the simplex)
let lr = 0.1;
agent.natural_gradient_update(&likelihood, lr);

let belief = agent.belief();
println!("Updated belief: {:?}", belief);

// Compute Fisher information matrix at current belief
let fisher = agent.fisher_information_matrix();
println!("Fisher diagonal: {:?}", fisher.diagonal());

// Fisher-Rao distance to another distribution
let other = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
let dist = agent.fisher_rao_distance(&other);
println!("Fisher-Rao distance: {:.4}", dist);
```

## API Reference

### `BeliefAgent`

| Method | Description |
|--------|-------------|
| `new(n_states)` | Create agent with uniform prior over n states. |
| `set_belief(p)` | Set belief state (probability vector). |
| `belief()` | Get current belief. |
| `natural_gradient_update(likelihood, lr)` | Update belief using F⁻¹∇. |
| `fisher_information_matrix()` | Compute Fisher matrix at current belief. |
| `fisher_rao_distance(other)` | Geodesic distance to another distribution. |
| `kl_divergence(other)` | KL(p ‖ q). |
| `entropy()` | Shannon entropy of belief. |

### `FisherRaoMetric`

| Method | Description |
|--------|-------------|
| `distance(p, q)` | Fisher-Rao geodesic distance. |
| `inner_product(p, u, v)` | Fisher inner product at p for tangent vectors u, v. |

### `AmariConnection`

| Method | Description |
|--------|-------------|
| `new(alpha)` | Create α-connection (0 = Levi-Civita, ±1 = exponential/expected). |
| `christoffel_symbols(p)` | Christoffel symbols at point p. |

### `NaturalGradient`

| Method | Description |
|--------|-------------|
| `compute(fisher, gradient)` | Compute F⁻¹∇. |
| `curvature_adaptive_step(fisher)` | Adaptive learning rate based on Fisher curvature. |

## How It Works

1. **Belief Representation**: Agent beliefs live on the probability simplex (positive, sum to 1).
2. **Fisher Metric**: g_ij(p) = E[∂log p/∂θᵢ · ∂log p/∂θⱼ]. This is the curvature of KL divergence.
3. **Natural Gradient**: ∇̃ = F⁻¹∇. Preconditioning by Fisher makes updates invariant to parameterization.
4. **α-Connections**: Amari's one-parameter family of affine connections. α=0 is metric-compatible (Levi-Civita), α=±1 are dually flat.
5. **Curvature-Adaptive Learning**: When Fisher eigenvalues are large (high curvature), step size decreases automatically.

## The Math

- **Fisher Information Matrix**: I(θ)ᵢⱼ = E_p[∂ᵢ log p(x|θ) · ∂ⱼ log p(x|θ)]
- **Natural Gradient**: ∇̃ℓ(θ) = I(θ)⁻¹∇ℓ(θ)
- **Fisher-Rao Distance**: d(p,q) = 2 arccos(Σ√pᵢqᵢ) for discrete distributions
- **KL Divergence**: D_KL(p‖q) = Σ pᵢ log(pᵢ/qᵢ)
- **Amari α-connection**: Γ⁽α⁾ᵢⱼₖ = E[(∂ᵢ∂ⱼ log p)(∂ₖ log p)] + (1-α)/2 · Tᵢⱼₖ

## Testing

118 tests covering:
- Fisher information matrix computation
- Natural gradient descent convergence
- Fisher-Rao distance properties (non-negativity, symmetry, triangle inequality)
- KL divergence computation
- Amari α-connection Christoffel symbols
- Belief agent update correctness
- Curvature-adaptive learning rates
- Entropy computation

## License

MIT
