//! Jeffreys prior: the uniform measure in the Fisher information metric.
//!
//! The Jeffreys prior is π(θ) ∝ √det(g(θ))
//! It is the unique prior that is invariant under reparameterization.

use crate::*;
use nalgebra::DMatrix;

/// Compute the Jeffreys prior density at θ.
/// π(θ) = √det(g(θ)) / Z
/// where Z = ∫ √det(g(θ)) dθ is the normalization constant.
#[derive(Debug, Clone)]
pub struct JeffreysPrior {
    /// Log of the normalization constant
    pub log_normalizer: f64,
}

impl JeffreysPrior {
    /// Compute the unnormalized Jeffreys prior: √det(g(θ))
    pub fn unnormalized(manifold: &dyn StatisticalManifold, theta: &ManifoldPoint) -> f64 {
        let g = manifold.fisher_information(theta);
        let det = g.determinant();
        if det > 0.0 {
            det.sqrt()
        } else {
            0.0
        }
    }

    /// Compute the log of the unnormalized Jeffreys prior: 1/2 log(det(g(θ)))
    pub fn log_unnormalized(manifold: &dyn StatisticalManifold, theta: &ManifoldPoint) -> f64 {
        let g = manifold.fisher_information(theta);
        0.5 * g.determinant().max(1e-300).ln()
    }

    /// Estimate the normalization constant via numerical integration.
    /// For 1D manifolds, integrate over a range.
    /// For 2D manifolds, integrate over a grid.
    pub fn normalize_1d(
        manifold: &dyn StatisticalManifold,
        theta_min: f64,
        theta_max: f64,
        n_samples: usize,
    ) -> Self {
        let dtheta = (theta_max - theta_min) / n_samples as f64;
        let mut integral = 0.0;
        for i in 0..n_samples {
            let theta = ManifoldPoint::new(vec![theta_min + (i as f64 + 0.5) * dtheta]);
            if manifold.valid_params(&theta) {
                integral += Self::unnormalized(manifold, &theta) * dtheta;
            }
        }
        Self {
            log_normalizer: integral.ln(),
        }
    }

    /// Estimate the normalization constant for 2D manifolds via grid integration.
    pub fn normalize_2d(
        manifold: &dyn StatisticalManifold,
        theta1_range: (f64, f64),
        theta2_range: (f64, f64),
        n_samples: usize,
    ) -> Self {
        let d1 = (theta1_range.1 - theta1_range.0) / n_samples as f64;
        let d2 = (theta2_range.1 - theta2_range.0) / n_samples as f64;
        let mut integral = 0.0;
        for i in 0..n_samples {
            for j in 0..n_samples {
                let t1 = theta1_range.0 + (i as f64 + 0.5) * d1;
                let t2 = theta2_range.0 + (j as f64 + 0.5) * d2;
                let theta = ManifoldPoint::new(vec![t1, t2]);
                if manifold.valid_params(&theta) {
                    integral += Self::unnormalized(manifold, &theta) * d1 * d2;
                }
            }
        }
        Self {
            log_normalizer: integral.ln(),
        }
    }

    /// Compute the normalized Jeffreys prior density.
    pub fn density(
        &self,
        manifold: &dyn StatisticalManifold,
        theta: &ManifoldPoint,
    ) -> f64 {
        let log_unnorm = Self::log_unnormalized(manifold, theta);
        (log_unnorm - self.log_normalizer).exp()
    }

    /// Sample from Jeffreys prior (approximate, via inverse CDF for 1D).
    /// Returns parameters sampled proportional to √det(g(θ)).
    pub fn sample_1d(
        manifold: &dyn StatisticalManifold,
        theta_min: f64,
        theta_max: f64,
        n_grid: usize,
        u: f64, // uniform random in [0, 1]
    ) -> ManifoldPoint {
        let dtheta = (theta_max - theta_min) / n_grid as f64;
        let mut cdf = Vec::with_capacity(n_grid);
        let mut total = 0.0;
        for i in 0..n_grid {
            let theta = ManifoldPoint::new(vec![theta_min + (i as f64 + 0.5) * dtheta]);
            let w = if manifold.valid_params(&theta) {
                Self::unnormalized(manifold, &theta)
            } else {
                0.0
            };
            total += w * dtheta;
            cdf.push(total);
        }

        let target = u * total;
        for (i, &c) in cdf.iter().enumerate() {
            if c >= target {
                return ManifoldPoint::new(vec![theta_min + (i as f64 + 0.5) * dtheta]);
            }
        }
        ManifoldPoint::new(vec![theta_max])
    }
}

/// Verify Jeffreys prior invariance under reparameterization.
/// If θ → φ = f(θ), then π(φ) = π(θ) |dθ/dφ|.
pub fn verify_reparameterization_invariance(
    manifold: &dyn StatisticalManifold,
    theta: &ManifoldPoint,
    jacobian: &DMatrix<f64>, // dθ/dφ
) -> bool {
    let g_theta = manifold.fisher_information(theta);
    let det_g_theta = g_theta.determinant();

    // Under reparameterization, g_φ = J^T g_θ J
    let g_phi = jacobian.transpose() * &g_theta * jacobian;
    let det_g_phi = g_phi.determinant();

    // Jeffreys prior transforms as: √det(g_φ) = √det(g_θ) · |det(J)|
    let jac_det = jacobian.determinant();
    let expected = det_g_theta * jac_det * jac_det;

    (det_g_phi - expected).abs() < 1e-6 * expected.abs().max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_jeffreys_unnormalized_positive() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let pi = JeffreysPrior::unnormalized(&m, &theta);
        assert!(pi > 0.0);
    }

    #[test]
    fn test_jeffreys_exponential_constant() {
        let m = ExponentialManifold;
        let t1 = ExponentialManifold::params(1.0);
        let t2 = ExponentialManifold::params(2.0);
        let pi1 = JeffreysPrior::unnormalized(&m, &t1);
        let pi2 = JeffreysPrior::unnormalized(&m, &t2);
        // Fisher for log λ is 1, so det = 1, √1 = 1 everywhere
        assert_relative_eq!(pi1, pi2, max_relative = 1e-10);
        assert_relative_eq!(pi1, 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_jeffreys_normalize_1d() {
        let m = ExponentialManifold;
        let prior = JeffreysPrior::normalize_1d(&m, 0.1, 5.0, 1000);
        // √(1/θ²) integrated from 0.1 to 5 = ln(5/0.1) = ln(50)
        assert!(prior.log_normalizer > 0.0);
    }

    #[test]
    fn test_jeffreys_density_integrates_to_1() {
        let m = ExponentialManifold;
        let prior = JeffreysPrior::normalize_1d(&m, -5.0, 5.0, 1000);
        let dx = 0.01;
        let mut integral = 0.0;
        let mut x = -5.0;
        while x < 5.0 {
            let theta = ManifoldPoint::new(vec![x]);
            integral += prior.density(&m, &theta) * dx;
            x += dx;
        }
        assert_relative_eq!(integral, 1.0, max_relative = 0.05);
    }

    #[test]
    fn test_jeffreys_sample_1d() {
        let m = ExponentialManifold;
        let sample = JeffreysPrior::sample_1d(&m, -5.0, 5.0, 1000, 0.5);
        // For uniform Jeffreys, median sample should be at midpoint
        assert!(sample.theta[0] > -1.0 && sample.theta[0] < 1.0);
    }

    #[test]
    fn test_jeffreys_normal_varies() {
        let m = NormalManifold;
        let t1 = NormalManifold::params(0.0, 1.0);
        let t2 = NormalManifold::params(0.0, 2.0);
        let pi1 = JeffreysPrior::unnormalized(&m, &t1);
        let pi2 = JeffreysPrior::unnormalized(&m, &t2);
        // Different σ should give different Jeffreys prior values
        assert!((pi1 - pi2).abs() > 1e-10);
    }

    #[test]
    fn test_reparameterization_invariance() {
        let m = ExponentialManifold;
        let theta = ExponentialManifold::params(2.0);
        // Jacobian of identity transformation
        let j = DMatrix::from_row_slice(1, 1, &[1.0]);
        assert!(verify_reparameterization_invariance(&m, &theta, &j));
    }

    #[test]
    fn test_jeffreys_beta_manifold() {
        let m = BetaManifold;
        let theta = BetaManifold::params(2.0, 3.0);
        let pi = JeffreysPrior::unnormalized(&m, &theta);
        assert!(pi > 0.0);
    }

    #[test]
    fn test_jeffreys_2d_normal() {
        let m = NormalManifold;
        let prior = JeffreysPrior::normalize_2d(&m, (-5.0, 5.0), (-3.0, 3.0), 100);
        assert!(prior.log_normalizer.is_finite());
    }
}
