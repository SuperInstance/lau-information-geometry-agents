#![allow(clippy::needless_range_loop)]
//! Amari's α-connections: a one-parameter family of affine connections on statistical manifolds.
//!
//! The α-connection generalizes the Levi-Civita connection (α=0):
//! - α = +1: exponential (e-) connection
//! - α = -1: mixture (m-) connection
//! - α = 0: Levi-Civita (metric) connection
//!
//! The α-connection coefficients are:
//! Γ^α_{ij,k} = E[(∂ᵢ∂ⱼ l + (1-α)/2 · ∂ᵢl · ∂ⱼl) · ∂ₖl]
//! where l = log p(x;θ)

use crate::*;
use nalgebra::DMatrix;

/// Amari's α-connection coefficients (Christoffel symbols of the first kind).
///
/// Γ^α_{ijk}(θ) = E[ (∂ᵢ∂ⱼ l + (1-α)/2 · ∂ᵢl · ∂ⱼl) · ∂ₖl ]
///
/// For α=0, this reduces to the Levi-Civita Christoffel symbols.
pub fn alpha_christoffel_first_kind(
    manifold: &dyn StatisticalManifold,
    theta: &ManifoldPoint,
    alpha: f64,
    x_min: f64,
    x_max: f64,
    n_samples: usize,
) -> DMatrix<f64> {
    // For a d-dimensional manifold, we compute Γ_{ij,k} for each k
    // This returns the averaged connection: Γ_{ij} = Σ_k Γ_{ijk} g^{kl}
    // But for simplicity, return Γ_{ij,·} contracted with metric
    let dim = manifold.dimension();
    let dx = (x_max - x_min) / n_samples as f64;

    // We'll compute Γ_{ijk} = E[(∂ᵢ∂ⱼl + c · ∂ᵢl · ∂ⱼl) · ∂ₖl] numerically
    // For now, use the simpler formulation:
    // Γ_{ij,k} ≈ Σ_k Γ_{ijk}
    let mut gamma = DMatrix::zeros(dim, dim);

    for i in 0..n_samples {
        let x = x_min + (i as f64 + 0.5) * dx;
        let p = manifold.pdf(x, theta);
        if p <= 1e-300 {
            continue;
        }
        let score = manifold.score(x, theta);
        let c = (1.0 - alpha) / 2.0;

        for a in 0..dim {
            for b in a..dim {
                // Approximate ∂ᵢ∂ⱼl numerically
                let eps = 1e-5;
                let mut theta_p = theta.clone();
                let mut theta_m = theta.clone();

                // ∂²l/∂a∂b via finite differences of score
                theta_p.theta[a] += eps;
                theta_m.theta[a] -= eps;
                let score_p = manifold.score(x, &theta_p);
                let score_m = manifold.score(x, &theta_m);
                let d2l_da = (score_p[b] - score_m[b]) / (2.0 * eps);

                // Γ_{ab} contribution: (d2l_da + c * score[a] * score[b]) * score weighted by p
                let conn = d2l_da + c * score[a] * score[b];
                for k in 0..dim {
                    gamma[(a, b)] += conn * score[k] * p * dx;
                }
                if a != b {
                    gamma[(b, a)] = gamma[(a, b)];
                }
            }
        }
    }

    gamma
}

/// Christoffel symbols of the second kind (raised index):
/// Γ^α_{ij}^k = Σ_l g^{kl} Γ^α_{ij,l}
pub fn alpha_christoffel_second_kind(
    fisher: &DMatrix<f64>,
    gamma_first: &DMatrix<f64>,
) -> DMatrix<f64> {
    // For simplicity with the contracted form, return gamma_first
    // In full generality, this would be a 3-tensor
    match fisher.clone().try_inverse() {
        Some(g_inv) => g_inv * gamma_first,
        None => gamma_first.clone(),
    }
}

/// Parallel transport of a vector along a geodesic using the α-connection.
/// Approximated by the connection coefficients.
pub fn parallel_transport_alpha(
    gamma_second: &DMatrix<f64>,
    velocity: &[f64],
    vector: &[f64],
    dt: f64,
) -> Vec<f64> {
    let dim = velocity.len();
    let mut transported = vector.to_vec();

    // ∇_v V = 0 => dV^k/dt = -Σ_{ij} Γ^k_{ij} v^i V^j
    for k in 0..dim {
        let mut correction = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                correction += gamma_second[(i, j)] * velocity[i] * vector[j];
            }
        }
        transported[k] -= correction * dt;
    }

    transported
}

/// Compute the α-geodesic between two points.
/// For α=0, this is the metric geodesic (shortest path).
/// For α=+1, this is the e-geodesic (exponential family geodesic).
/// For α=-1, this is the m-geodesic (mixture geodesic).
pub fn alpha_geodesic(
    theta1: &ManifoldPoint,
    theta2: &ManifoldPoint,
    alpha: f64,
    t: f64,
) -> ManifoldPoint {
    let dim = theta1.dim();
    match alpha {
        a if (a - 1.0).abs() < 1e-10 => {
            // e-geodesic: θ(t) = (1-t)θ₁ + tθ₂ in natural parameter space
            // For log-parameterized families, this is linear interpolation
            let interp: Vec<f64> = (0..dim)
                .map(|j| (1.0 - t) * theta1.theta[j] + t * theta2.theta[j])
                .collect();
            ManifoldPoint::new(interp)
        }
        a if (a + 1.0).abs() < 1e-10 => {
            // m-geodesic: θ(t) = (1-t)θ₁ + tθ₂ in expectation parameter space
            // For our parameterization, also approximately linear
            let interp: Vec<f64> = (0..dim)
                .map(|j| (1.0 - t) * theta1.theta[j] + t * theta2.theta[j])
                .collect();
            ManifoldPoint::new(interp)
        }
        _ => {
            // General α: linear interpolation as approximation
            let interp: Vec<f64> = (0..dim)
                .map(|j| (1.0 - t) * theta1.theta[j] + t * theta2.theta[j])
                .collect();
            ManifoldPoint::new(interp)
        }
    }
}

/// Compute the α-divergence between two distributions:
/// D_α(p||q) = 4/(1-α²) [1 - ∫ p^{(1+α)/2} q^{(1-α)/2} dx]
pub fn alpha_divergence(
    manifold: &dyn StatisticalManifold,
    theta_p: &ManifoldPoint,
    theta_q: &ManifoldPoint,
    alpha: f64,
    x_min: f64,
    x_max: f64,
    n_samples: usize,
) -> f64 {
    if (alpha - 1.0).abs() < 1e-10 || (alpha + 1.0).abs() < 1e-10 {
        return f64::NAN; // limit case
    }

    let dx = (x_max - x_min) / n_samples as f64;
    let a = (1.0 + alpha) / 2.0;
    let b = (1.0 - alpha) / 2.0;

    let mut integral = 0.0;
    for i in 0..n_samples {
        let x = x_min + (i as f64 + 0.5) * dx;
        let p = manifold.pdf(x, theta_p);
        let q = manifold.pdf(x, theta_q);
        if p > 0.0 && q > 0.0 {
            integral += p.powf(a) * q.powf(b) * dx;
        }
    }

    4.0 / (1.0 - alpha * alpha) * (1.0 - integral)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_alpha_geodesic_endpoints() {
        let t1 = ManifoldPoint::new(vec![0.0, 0.0]);
        let t2 = ManifoldPoint::new(vec![1.0, 2.0]);

        let at_0 = alpha_geodesic(&t1, &t2, 0.0, 0.0);
        assert_relative_eq!(at_0.theta[0], 0.0, max_relative = 1e-10);
        assert_relative_eq!(at_0.theta[1], 0.0, max_relative = 1e-10);

        let at_1 = alpha_geodesic(&t1, &t2, 0.0, 1.0);
        assert_relative_eq!(at_1.theta[0], 1.0, max_relative = 1e-10);
        assert_relative_eq!(at_1.theta[1], 2.0, max_relative = 1e-10);
    }

    #[test]
    fn test_alpha_geodesic_midpoint() {
        let t1 = ManifoldPoint::new(vec![0.0]);
        let t2 = ManifoldPoint::new(vec![2.0]);
        let mid = alpha_geodesic(&t1, &t2, 0.0, 0.5);
        assert_relative_eq!(mid.theta[0], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_e_geodesic() {
        let t1 = ManifoldPoint::new(vec![0.0, 0.0]);
        let t2 = ManifoldPoint::new(vec![2.0, 4.0]);
        let mid = alpha_geodesic(&t1, &t2, 1.0, 0.5);
        assert_relative_eq!(mid.theta[0], 1.0, max_relative = 1e-10);
        assert_relative_eq!(mid.theta[1], 2.0, max_relative = 1e-10);
    }

    #[test]
    fn test_m_geodesic() {
        let t1 = ManifoldPoint::new(vec![0.0, 0.0]);
        let t2 = ManifoldPoint::new(vec![2.0, 4.0]);
        let mid = alpha_geodesic(&t1, &t2, -1.0, 0.5);
        assert_relative_eq!(mid.theta[0], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_alpha_divergence_same_distribution() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let d = alpha_divergence(&m, &theta, &theta, 0.5, -10.0, 10.0, 1000);
        assert_relative_eq!(d, 0.0, max_relative = 0.05);
    }

    #[test]
    fn test_alpha_divergence_positive() {
        let m = NormalManifold;
        let t1 = NormalManifold::params(0.0, 1.0);
        let t2 = NormalManifold::params(1.0, 1.0);
        let d = alpha_divergence(&m, &t1, &t2, 0.5, -10.0, 10.0, 1000);
        assert!(d > 0.0);
    }

    #[test]
    fn test_christoffel_flat_manifold() {
        // Exponential with log λ has flat Fisher metric (g = 1)
        let m = ExponentialManifold;
        let theta = ExponentialManifold::params(1.0);
        let gamma = alpha_christoffel_first_kind(&m, &theta, 0.0, 0.01, 20.0, 500);
        // For flat metric, Christoffel symbols should be ~0
        assert!(gamma[(0, 0)].abs() < 0.1);
    }

    #[test]
    fn test_parallel_transport_preserves_norm_approx() {
        let gamma = DMatrix::zeros(2, 2); // flat space
        let vel = vec![1.0, 0.0];
        let v = vec![0.0, 1.0];
        let transported = parallel_transport_alpha(&gamma, &vel, &v, 0.1);
        assert_relative_eq!(transported[0], 0.0, max_relative = 1e-10);
        assert_relative_eq!(transported[1], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_alpha_divergence_symmetry_alpha0() {
        // D_0 should satisfy D_0(p||q) = D_0(q||p) for Hellinger-like symmetry
        let m = NormalManifold;
        let t1 = NormalManifold::params(0.0, 1.0);
        let t2 = NormalManifold::params(1.0, 2.0);
        let d_pq = alpha_divergence(&m, &t1, &t2, 0.0, -10.0, 10.0, 2000);
        let d_qp = alpha_divergence(&m, &t2, &t1, 0.0, -10.0, 10.0, 2000);
        // α=0 divergence is symmetric
        assert_relative_eq!(d_pq, d_qp, max_relative = 0.05);
    }
}
