//! Fisher information metric computation and properties.

use crate::*;
use nalgebra::DMatrix;

/// Compute the Fisher information matrix via numerical integration of the score.
/// g_ij(θ) = ∫ ∂ᵢ log p(x;θ) · ∂ⱼ log p(x;θ) · p(x;θ) dx
pub fn fisher_metric_numerical(
    manifold: &dyn StatisticalManifold,
    theta: &ManifoldPoint,
    x_min: f64,
    x_max: f64,
    n_samples: usize,
) -> DMatrix<f64> {
    let dim = manifold.dimension();
    let dx = (x_max - x_min) / n_samples as f64;
    let mut g = DMatrix::zeros(dim, dim);

    for i in 0..n_samples {
        let x = x_min + (i as f64 + 0.5) * dx;
        let p = manifold.pdf(x, theta);
        if p <= 0.0 {
            continue;
        }
        let score = manifold.score(x, theta);
        for a in 0..dim {
            for b in a..dim {
                g[(a, b)] += score[a] * score[b] * p * dx;
            }
        }
    }
    // Symmetrize
    for a in 0..dim {
        for b in (a + 1)..dim {
            g[(b, a)] = g[(a, b)];
        }
    }
    g
}

/// Verify Fisher information is symmetric positive semi-definite.
pub fn is_positive_definite(g: &DMatrix<f64>) -> bool {
    let n = g.nrows();
    if g.nrows() != g.ncols() {
        return false;
    }
    // Check symmetric
    for i in 0..n {
        for j in (i + 1)..n {
            if (g[(i, j)] - g[(j, i)]).abs() > 1e-10 {
                return false;
            }
        }
    }
    // Check all eigenvalues > 0 via leading principal minors
    // Simple check: try Cholesky
    g.clone().cholesky().is_some()
}

/// Compute the volume element: sqrt(det(g)) — the Riemannian volume density.
pub fn volume_element(g: &DMatrix<f64>) -> f64 {
    g.determinant().abs().sqrt()
}

/// Metric tensor trace (total Fisher information content).
pub fn fisher_trace(g: &DMatrix<f64>) -> f64 {
    let n = g.nrows();
    (0..n).map(|i| g[(i, i)]).sum()
}

/// Fisher-Rao information matrix condition number.
pub fn fisher_condition_number(g: &DMatrix<f64>) -> f64 {
    let svd = g.clone().svd(false, false);
    let s = svd.singular_values;
    let max_s = s.max();
    let min_s = s.min();
    if min_s > 1e-15 {
        max_s / min_s
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fisher_metric_numerical_normal() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let g_num = fisher_metric_numerical(&m, &theta, -10.0, 10.0, 10000);
        let g_exact = m.fisher_information(&theta);
        assert_relative_eq!(g_num[(0, 0)], g_exact[(0, 0)], max_relative = 0.05);
        assert_relative_eq!(g_num[(1, 1)], g_exact[(1, 1)], max_relative = 0.05);
    }

    #[test]
    fn test_is_positive_definite() {
        let g = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        assert!(is_positive_definite(&g));
    }

    #[test]
    fn test_not_positive_definite() {
        let g = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]);
        assert!(!is_positive_definite(&g));
    }

    #[test]
    fn test_volume_element_identity() {
        let g = DMatrix::identity(2, 2);
        assert_relative_eq!(volume_element(&g), 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_volume_element_normal() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let g = m.fisher_information(&theta);
        let vol = volume_element(&g);
        // det = 1 * 2 = 2, sqrt = sqrt(2)
        assert_relative_eq!(vol, 2.0_f64.sqrt(), max_relative = 1e-10);
    }

    #[test]
    fn test_fisher_trace_normal() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let g = m.fisher_information(&theta);
        assert_relative_eq!(fisher_trace(&g), 3.0, max_relative = 1e-10); // 1 + 2
    }

    #[test]
    fn test_fisher_condition_well_conditioned() {
        let g = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let cn = fisher_condition_number(&g);
        assert_relative_eq!(cn, 2.0, max_relative = 1e-10);
    }

    #[test]
    fn test_normal_fisher_is_pd() {
        let m = NormalManifold;
        let theta = NormalManifold::params(2.0, 3.0);
        let g = m.fisher_information(&theta);
        assert!(is_positive_definite(&g));
    }

    #[test]
    fn test_exponential_fisher_is_pd() {
        let m = ExponentialManifold;
        let theta = ExponentialManifold::params(1.0);
        let g = m.fisher_information(&theta);
        assert!(is_positive_definite(&g));
    }
}
