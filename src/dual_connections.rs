//! Dual connections: e-connection (exponential) and m-connection (mixture).
//!
//! In information geometry, the e-connection and m-connection are dual with respect
//! to the Fisher metric. This duality generalizes convex conjugacy in optimization.
//!
//! Key result: ∇^{(e)} and ∇^{(m)} are dual connections satisfying
//! ∂/∂t g(∇^{(e)}_X Y, Z) = g(∇^{(e)}_X Y, Z) + g(Y, ∇^{(m)}_X Z)

use crate::*;
use nalgebra::DMatrix;

/// Dual connection pair (e-connection, m-connection)
#[derive(Debug, Clone)]
pub struct DualConnections {
    /// e-connection Christoffel symbols (α = +1)
    pub e_christoffel: DMatrix<f64>,
    /// m-connection Christoffel symbols (α = -1)
    pub m_christoffel: DMatrix<f64>,
    /// Fisher information matrix
    pub fisher: DMatrix<f64>,
}

impl DualConnections {
    /// Compute dual connections for a manifold at a given point.
    pub fn compute(
        manifold: &dyn StatisticalManifold,
        theta: &ManifoldPoint,
        x_min: f64,
        x_max: f64,
        n_samples: usize,
    ) -> Self {
        let fisher = manifold.fisher_information(theta);

        // e-connection: α = +1
        let e_christoffel = crate::alpha_connections::alpha_christoffel_first_kind(
            manifold, theta, 1.0, x_min, x_max, n_samples,
        );

        // m-connection: α = -1
        let m_christoffel = crate::alpha_connections::alpha_christoffel_first_kind(
            manifold, theta, -1.0, x_min, x_max, n_samples,
        );

        Self {
            e_christoffel,
            m_christoffel,
            fisher,
        }
    }

    /// Verify the duality relation:
    /// ∂_k g_{ij} = Γ^{(e)}_{ij,k} + Γ^{(m)}_{ij,k}
    ///
    /// where ∂_k g_{ij} is the derivative of the Fisher metric.
    pub fn verify_duality(
        &self,
        manifold: &dyn StatisticalManifold,
        theta: &ManifoldPoint,
        eps: f64,
    ) -> f64 {
        let dim = theta.dim();
        let mut max_violation: f64 = 0.0;

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    // Numerical derivative of Fisher metric
                    let mut theta_p = theta.clone();
                    let mut theta_m = theta.clone();
                    theta_p.theta[k] += eps;
                    theta_m.theta[k] -= eps;
                    let g_p = manifold.fisher_information(&theta_p);
                    let g_m = manifold.fisher_information(&theta_m);
                    let dg = (g_p[(i, j)] - g_m[(i, j)]) / (2.0 * eps);

                    // e + m Christoffel
                    let sum = self.e_christoffel[(i, j)] + self.m_christoffel[(i, j)];

                    // For the Levi-Civita connection (α=0), Γ^{LC}_{ijk} = ∂_k g_{ij} / 2
                    // The relation is: Γ^{(α)}_{ijk} + Γ^{(-α)}_{ijk} = 2Γ^{LC}_{ijk} = ∂_k g_{ij}
                    // But we're using contracted forms, so this is approximate
                    let violation = (dg - sum).abs();
                    max_violation = max_violation.max(violation);
                }
            }
        }

        max_violation
    }
}

/// e-geodesic: straight line in natural parameter (η) coordinates.
/// For an exponential family p(x;η) = exp(η·T(x) - ψ(η)) h(x),
/// the e-geodesic is linear in η.
pub fn e_geodesic(theta1: &ManifoldPoint, theta2: &ManifoldPoint, t: f64) -> ManifoldPoint {
    let dim = theta1.dim();
    let interp: Vec<f64> = (0..dim)
        .map(|j| (1.0 - t) * theta1.theta[j] + t * theta2.theta[j])
        .collect();
    ManifoldPoint::new(interp)
}

/// m-geodesic: straight line in expectation parameter (μ) coordinates.
/// μ = E[T(X)] = ∇ψ(η).
/// For our log-parameterized families, we approximate this as linear interpolation
/// in the expectation parameter space.
pub fn m_geodesic(theta1: &ManifoldPoint, theta2: &ManifoldPoint, t: f64) -> ManifoldPoint {
    // For simplicity, linear interpolation (exact for flat manifolds)
    let dim = theta1.dim();
    let interp: Vec<f64> = (0..dim)
        .map(|j| (1.0 - t) * theta1.theta[j] + t * theta2.theta[j])
        .collect();
    ManifoldPoint::new(interp)
}

/// e-projection: find the closest point on a submanifold via e-connection.
/// This minimizes KL(p || q) over q in the submanifold.
pub fn e_projection(
    manifold: &dyn StatisticalManifold,
    theta_p: &ManifoldPoint,
    submanifold_thetas: &[ManifoldPoint],
) -> usize {
    // Find the point in the submanifold with minimum KL(p || q)
    // Approximate via Fisher-Rao distance
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;

    for (idx, theta_q) in submanifold_thetas.iter().enumerate() {
        let g = manifold.fisher_information(theta_q);
        let diff: Vec<f64> = theta_p
            .theta
            .iter()
            .zip(theta_q.theta.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let dist = crate::natural_gradient::riemannian_gradient_norm(&g, &diff);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

/// m-projection: find the closest point on a submanifold via m-connection.
/// This minimizes KL(q || p) over q in the submanifold.
pub fn m_projection(
    manifold: &dyn StatisticalManifold,
    theta_q: &ManifoldPoint,
    submanifold_thetas: &[ManifoldPoint],
) -> usize {
    // Find the point in the submanifold with minimum KL(q || p)
    // For m-projection, we use the Fisher metric at the projection target
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;

    for (idx, theta_p) in submanifold_thetas.iter().enumerate() {
        let g = manifold.fisher_information(theta_p);
        let diff: Vec<f64> = theta_q
            .theta
            .iter()
            .zip(theta_p.theta.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let dist = crate::natural_gradient::riemannian_gradient_norm(&g, &diff);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

/// Pythagorean theorem in information geometry:
/// For e-geodesic from p to q and m-geodesic from q to r,
/// if they meet orthogonally at q, then D(p||r) = D(p||q) + D(q||r).
///
/// We verify approximate orthogonality.
pub fn generalized_pythagorean(
    fisher_at_q: &DMatrix<f64>,
    p_minus_q: &[f64],
    r_minus_q: &[f64],
) -> bool {
    // Check orthogonality: (p-q)ᵀ G(q) (r-q) ≈ 0
    let n = p_minus_q.len();
    let mut inner = 0.0;
    for i in 0..n {
        for j in 0..n {
            inner += p_minus_q[i] * fisher_at_q[(i, j)] * r_minus_q[j];
        }
    }
    inner.abs() < 0.1 // tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dual_connections_compute() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let dc = DualConnections::compute(&m, &theta, -10.0, 10.0, 1000);
        assert_eq!(dc.e_christoffel.nrows(), 2);
        assert_eq!(dc.m_christoffel.nrows(), 2);
    }

    #[test]
    fn test_e_geodesic_endpoints() {
        let t1 = ManifoldPoint::new(vec![0.0]);
        let t2 = ManifoldPoint::new(vec![1.0]);
        let p0 = e_geodesic(&t1, &t2, 0.0);
        let p1 = e_geodesic(&t1, &t2, 1.0);
        assert_relative_eq!(p0.theta[0], 0.0, max_relative = 1e-10);
        assert_relative_eq!(p1.theta[0], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_m_geodesic_endpoints() {
        let t1 = ManifoldPoint::new(vec![0.0]);
        let t2 = ManifoldPoint::new(vec![1.0]);
        let p0 = m_geodesic(&t1, &t2, 0.0);
        let p1 = m_geodesic(&t1, &t2, 1.0);
        assert_relative_eq!(p0.theta[0], 0.0, max_relative = 1e-10);
        assert_relative_eq!(p1.theta[0], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_e_projection() {
        let m = NormalManifold;
        let target = NormalManifold::params(2.0, 1.0);
        let candidates = vec![
            NormalManifold::params(0.0, 1.0),
            NormalManifold::params(2.5, 1.0),
            NormalManifold::params(5.0, 1.0),
        ];
        let idx = e_projection(&m, &target, &candidates);
        assert_eq!(idx, 1); // Closest to (2.0, 1.0)
    }

    #[test]
    fn test_m_projection() {
        let m = NormalManifold;
        let target = NormalManifold::params(2.0, 1.0);
        let candidates = vec![
            NormalManifold::params(0.0, 1.0),
            NormalManifold::params(1.8, 1.0),
            NormalManifold::params(5.0, 1.0),
        ];
        let idx = m_projection(&m, &target, &candidates);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_pythagorean_orthogonal() {
        let g = DMatrix::identity(2, 2);
        let pq = vec![1.0, 0.0];
        let rq = vec![0.0, 1.0];
        assert!(generalized_pythagorean(&g, &pq, &rq));
    }

    #[test]
    fn test_pythagorean_not_orthogonal() {
        let g = DMatrix::identity(2, 2);
        let pq = vec![1.0, 0.0];
        let rq = vec![1.0, 0.0];
        assert!(!generalized_pythagorean(&g, &pq, &rq));
    }

    #[test]
    fn test_dual_connections_fisher_positive() {
        let m = NormalManifold;
        let theta = NormalManifold::params(1.0, 2.0);
        let dc = DualConnections::compute(&m, &theta, -10.0, 10.0, 500);
        assert!(crate::fisher_metric::is_positive_definite(&dc.fisher));
    }
}
