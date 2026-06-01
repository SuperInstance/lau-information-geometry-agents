//! Natural gradient: steepest descent in the Fisher information metric.
//!
//! The natural gradient at θ is: ∇̃f = G⁻¹ ∇f
//! where G is the Fisher information matrix and ∇f is the Euclidean gradient.

use crate::*;
use nalgebra::DMatrix;

/// Compute the natural gradient: G⁻¹ · ∇L
///
/// In Euclidean space, the steepest descent direction is -∇L.
/// On the statistical manifold with Fisher metric G, the steepest descent is:
///   -G⁻¹ ∇L
///
/// This respects the geometry of the probability simplex.
pub fn natural_gradient(
    fisher: &DMatrix<f64>,
    euclidean_grad: &[f64],
) -> Vec<f64> {
    let n = euclidean_grad.len();
    let g = DVector::from_vec(euclidean_grad.to_vec());

    // Try to invert Fisher matrix
    match fisher.clone().try_inverse() {
        Some(g_inv) => {
            let ng = g_inv * g;
            ng.iter().cloned().collect()
        }
        None => {
            // Regularize: G + εI
            let eps = 1e-6;
            let mut regularized = fisher.clone();
            for i in 0..n {
                regularized[(i, i)] += eps;
            }
            match regularized.try_inverse() {
                Some(g_inv) => {
                    let ng = g_inv * g;
                    ng.iter().cloned().collect()
                }
                None => {
                    // Fallback: scale by 1/diagonal
                    (0..n).map(|i| euclidean_grad[i] / (fisher[(i, i)] + eps)).collect()
                }
            }
        }
    }
}

/// Natural gradient descent step.
/// θ_{t+1} = θ_t - η · G⁻¹(θ_t) · ∇L(θ_t)
pub fn natural_gradient_step(
    manifold: &dyn StatisticalManifold,
    theta: &ManifoldPoint,
    euclidean_grad: &[f64],
    learning_rate: f64,
) -> ManifoldPoint {
    let fisher = manifold.fisher_information(theta);
    let nat_grad = natural_gradient(&fisher, euclidean_grad);
    let new_theta: Vec<f64> = theta
        .theta
        .iter()
        .zip(nat_grad.iter())
        .map(|(&t, &g)| t - learning_rate * g)
        .collect();
    ManifoldPoint::new(new_theta)
}

/// Compute the Riemannian gradient magnitude: √(gᵀ G g)
pub fn riemannian_gradient_norm(
    fisher: &DMatrix<f64>,
    grad: &[f64],
) -> f64 {
    let n = grad.len();
    let _g = DVector::from_vec(grad.to_vec());
    let mut quad = 0.0;
    for i in 0..n {
        for j in 0..n {
            quad += grad[i] * fisher[(i, j)] * grad[j];
        }
    }
    quad.max(0.0).sqrt()
}

/// Compute the KL divergence gradient approximation.
/// ∇KL(p_θ || p_θ') ≈ Fisher information × (θ - θ') for nearby θ.
pub fn kl_gradient_approx(
    fisher: &DMatrix<f64>,
    theta: &ManifoldPoint,
    theta_ref: &ManifoldPoint,
) -> Vec<f64> {
    let diff: Vec<f64> = theta
        .theta
        .iter()
        .zip(theta_ref.theta.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let d = DVector::from_vec(diff);
    let result = fisher * d;
    result.iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_natural_gradient_identity() {
        let g = DMatrix::identity(2, 2);
        let grad = vec![1.0, 2.0];
        let ng = natural_gradient(&g, &grad);
        assert_relative_eq!(ng[0], 1.0, max_relative = 1e-10);
        assert_relative_eq!(ng[1], 2.0, max_relative = 1e-10);
    }

    #[test]
    fn test_natural_gradient_scaled() {
        let g = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 4.0]);
        let grad = vec![4.0, 8.0];
        let ng = natural_gradient(&g, &grad);
        assert_relative_eq!(ng[0], 1.0, max_relative = 1e-6);
        assert_relative_eq!(ng[1], 2.0, max_relative = 1e-6);
    }

    #[test]
    fn test_natural_gradient_step() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let grad = vec![1.0, 0.5];
        let new_theta = natural_gradient_step(&m, &theta, &grad, 0.1);
        // New μ should move in negative gradient direction, scaled by Fisher inverse
        assert!(new_theta.theta[0] < theta.theta[0]);
    }

    #[test]
    fn test_natural_gradient_respects_geometry() {
        // Natural gradient should be shorter in high-information directions
        let g = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 1.0]);
        let grad = vec![1.0, 1.0];
        let ng = natural_gradient(&g, &grad);
        // Direction 0 has high Fisher info → natural gradient should be smaller
        assert!(ng[0].abs() < ng[1].abs());
    }

    #[test]
    fn test_riemannian_gradient_norm() {
        let g = DMatrix::identity(2, 2);
        let grad = vec![3.0, 4.0];
        let norm = riemannian_gradient_norm(&g, &grad);
        assert_relative_eq!(norm, 5.0, max_relative = 1e-10);
    }

    #[test]
    fn test_riemannian_norm_with_metric() {
        let g = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 1.0]);
        let grad = vec![1.0, 0.0];
        let norm = riemannian_gradient_norm(&g, &grad);
        assert_relative_eq!(norm, 2.0, max_relative = 1e-10);
    }

    #[test]
    fn test_kl_gradient_approx_zero_at_same_point() {
        let m = NormalManifold;
        let theta = NormalManifold::params(1.0, 2.0);
        let fisher = m.fisher_information(&theta);
        let grad = kl_gradient_approx(&fisher, &theta, &theta);
        assert_relative_eq!(grad[0], 0.0, max_relative = 1e-10);
        assert_relative_eq!(grad[1], 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_natural_gradient_regularization() {
        // Nearly singular Fisher matrix
        let g = DMatrix::from_row_slice(2, 2, &[1e-20, 0.0, 0.0, 1.0]);
        let grad = vec![1.0, 1.0];
        let ng = natural_gradient(&g, &grad);
        // Should still produce finite results via regularization
        assert!(ng[0].is_finite());
        assert!(ng[1].is_finite());
    }

    #[test]
    fn test_natural_gradient_exponential() {
        let m = ExponentialManifold;
        let theta = ExponentialManifold::params(2.0);
        let fisher = m.fisher_information(&theta);
        let grad = vec![3.0];
        let ng = natural_gradient(&fisher, &grad);
        // For exponential with log λ, Fisher = 1, so natural = Euclidean
        assert_relative_eq!(ng[0], 3.0, max_relative = 1e-10);
    }

    #[test]
    fn test_descent_decreases_norm() {
        let m = NormalManifold;
        let theta = NormalManifold::params(3.0, 1.0);
        let grad = vec![2.0, 0.0]; // gradient in μ direction
        let new_theta = natural_gradient_step(&m, &theta, &grad, 0.01);
        // μ should decrease
        assert!(new_theta.theta[0] < theta.theta[0]);
    }
}
