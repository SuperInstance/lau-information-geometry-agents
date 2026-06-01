//! # lau-information-geometry-agents
//!
//! Information geometry treats agent belief spaces as statistical manifolds equipped
//! with the Fisher information metric. Learning is geodesic motion, and the curvature
//! of the belief manifold determines how hard learning is.
//!
//! ## Core Concepts
//!
//! - **Fisher Information Metric**: The natural Riemannian metric on probability distributions
//! - **Fisher-Rao Distance**: Geodesic distance between belief states
//! - **Natural Gradient**: Steepest descent in the Fisher metric (not Euclidean)
//! - **Amari's α-connections**: Family of affine connections generalizing Levi-Civita
//! - **Dual Connections**: e-connection (exponential) and m-connection (mixture) are dual
//! - **Curvature of Belief Space**: Positive = easy learning, negative = hard learning
//! - **Jeffreys Prior**: Uniform measure in the Fisher metric
//! - **Chentsov's Theorem**: Fisher metric is the unique invariant metric on the probability simplex

pub mod manifold;
pub mod fisher_metric;
pub mod fisher_rao;
pub mod natural_gradient;
pub mod alpha_connections;
pub mod dual_connections;
pub mod curvature;
pub mod jeffreys_prior;
pub mod chentsov;
pub mod monotonicity;
pub mod agent_learning;

pub use manifold::*;
pub use fisher_metric::*;
pub use fisher_rao::*;
pub use natural_gradient::*;
pub use alpha_connections::*;
pub use dual_connections::*;
pub use curvature::*;
pub use jeffreys_prior::*;
pub use chentsov::*;
pub use monotonicity::*;
pub use agent_learning::*;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A point on a statistical manifold, parameterized by θ ∈ ℝⁿ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldPoint {
    /// Parameter vector θ
    pub theta: Vec<f64>,
}

impl ManifoldPoint {
    pub fn new(theta: Vec<f64>) -> Self {
        Self { theta }
    }

    pub fn dim(&self) -> usize {
        self.theta.len()
    }

    pub fn to_dvector(&self) -> DVector<f64> {
        DVector::from_vec(self.theta.clone())
    }
}

impl From<Vec<f64>> for ManifoldPoint {
    fn from(theta: Vec<f64>) -> Self {
        Self::new(theta)
    }
}

/// A parametric family of probability distributions forming a statistical manifold
pub trait StatisticalManifold: Send + Sync {
    /// Dimension of the parameter space
    fn dimension(&self) -> usize;

    /// Probability density p(x; θ)
    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64;

    /// Log-probability log p(x; θ)
    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        self.pdf(x, theta).ln()
    }

    /// Score function: ∂/∂θᵢ log p(x; θ)
    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64>;

    /// Fisher information matrix g_ij(θ) = E[∂ᵢ log p · ∂ⱼ log p]
    fn fisher_information(&self, theta: &ManifoldPoint) -> DMatrix<f64>;

    /// Sample from p(x; θ) — optional, default panics
    fn sample(&self, _theta: &ManifoldPoint, _rng: &mut dyn rand::RngCore) -> f64 {
        unimplemented!("Sampling not implemented for this manifold")
    }

    /// Check if parameters are valid
    fn valid_params(&self, theta: &ManifoldPoint) -> bool;
}

/// Convert Fisher info to nalgebra matrix
pub fn fisher_to_nalgebra(fisher: &[Vec<f64>]) -> DMatrix<f64> {
    let n = fisher.len();
    DMatrix::from_fn(n, n, |i, j| fisher[i][j])
}
