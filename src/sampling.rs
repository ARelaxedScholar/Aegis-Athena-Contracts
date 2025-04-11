use pyo3::prelude::*;

use nalgebra::base::dimension::Dyn;
use rand::{Rng, SeedableRng, distributions::Distribution, rngs::OsRng};
use rand_chacha::ChaCha20Rng; // <— new
use rand_distr::Uniform;
use statrs::distribution::MultivariateNormal;

#[pyclass]
pub struct PySampler {
    sampler: Sampler,
}

#[pymethods]
impl PySampler {
    /// Construct a new sampler: mode = "factor" or "normal", seed = Option<u64>
    #[new]
    #[pyo3(signature = (mode, assets, factors, periods, seed = None))]
    fn new(
        mode: &str,
        assets: usize,
        factors: usize,
        periods: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let sampler = match mode {
            "factor" => Sampler::factor_model_synthetic(assets, factors, periods, seed)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?,
            "normal" => {
                Sampler::normal(&vec![0.0; assets], &vec![1.0; assets * assets], periods, seed)
            }.map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "mode must be \"factor\" or \"normal\"",
                ));
            }
        };
        Ok(PySampler { sampler })
    }

    /// Sample returns; advances internal RNG
    fn sample_returns(&mut self) -> Vec<Vec<f64>> {
        self.sampler.sample_returns()
    }

    /// Reseed the internal RNG mid-flight
    fn reseed(&mut self, seed: u64) {
        self.sampler.reseed(seed);
    }
}

#[derive(Debug, Clone)]
pub enum Sampler {
    FactorModel {
        assets_under_management: usize,
        periods_to_sample: usize,
        number_of_factors: usize,

        mu_factors: Vec<f64>,
        covariance_factors: Vec<Vec<f64>>,
        loadings: Vec<Vec<f64>>,
        idiosyncratic_variances: Vec<f64>,

        mu_assets: Vec<f64>,
        covariance_assets: Vec<Vec<f64>>,

        // dynamic-dimension MVN
        normal_distribution: MultivariateNormal<Dyn>,
        rng: ChaCha20Rng,
    },

    Normal {
        periods_to_sample: usize,
        normal_distribution: MultivariateNormal<Dyn>,
        rng: ChaCha20Rng,
    },

    SeriesGAN(usize),
}

impl Sampler {
    pub fn factor_model_synthetic(
        assets_under_management: usize,
        number_of_factors: usize,
        periods_to_sample: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        if assets_under_management == 0 || number_of_factors == 0 {
            return Err("Assets and Factors should be positives".into());
        }

        let mut rng = if let Some(s) = seed {
            ChaCha20Rng::seed_from_u64(s)
        } else {
            ChaCha20Rng::from_entropy()
        };

        let small_returns = 0.001;
        let mu_factors = vec![small_returns; number_of_factors];
        let covariance_factors = Self::generate_covariance_matrix(number_of_factors)?;

        let throwaway = MultivariateNormal::new(
            mu_factors.clone(),
            covariance_factors.clone().into_iter().flatten().collect(),
        )
        .map_err(|e| format!("MVN init failed: {}", e))?;
        let factor_returns = throwaway.sample(&mut rng);

        let mut loadings = vec![vec![0.0; number_of_factors]; assets_under_management];
        let uniform = Uniform::new(0.5, 1.5);
        for i in 0..assets_under_management {
            for j in 0..number_of_factors {
                loadings[i][j] = rng.sample(uniform);
            }
        }

        let idiosyncratic_variances = vec![0.01; assets_under_management];

        let mut mu_assets = vec![0.0; assets_under_management];
        for i in 0..assets_under_management {
            mu_assets[i] = loadings[i]
                .iter()
                .zip(factor_returns.iter())
                .map(|(l, f)| l * f)
                .sum();
        }

        let covariance_assets = Self::compute_asset_covariance(
            &loadings,
            &covariance_factors,
            &idiosyncratic_variances,
        )?;

        let normal_distribution = MultivariateNormal::new(
            mu_assets.clone(),
            covariance_assets.clone().into_iter().flatten().collect(),
        )
        .map_err(|e| format!("Failed to create MVN: {}", e))?;

        Ok(Sampler::FactorModel {
            assets_under_management,
            periods_to_sample,
            number_of_factors,
            mu_factors,
            covariance_factors,
            loadings,
            idiosyncratic_variances,
            mu_assets,
            covariance_assets,
            normal_distribution,
            rng,
        })
    }

    pub fn run_batch_factor_model_maker(
        mu_assets: &[f64],
        covariance_assets: &[Vec<f64>],
        periods_to_sample: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let normal_sampler = Self::normal(mu_assets, &covariance_assets.iter().flatten().copied().collect::<Vec<f64>>(), periods_to_sample, seed)?;
        let (periods_to_sample, normal_distribution, rng) = {if let Self::Normal { periods_to_sample, normal_distribution, rng,} = normal_sampler {
            (periods_to_sample, normal_distribution, rng)
        } else {
            panic!("Schema of Normal Distribution sampler was changed without updating related function.")
        }};
        
        Ok(Sampler::FactorModel {
            // we don't care since the important info is passed by the asset_means implicitly
            // or is only useful for logging/construction
            assets_under_management: mu_assets.len(),
            number_of_factors: 0,
            mu_factors: vec![0.],
            covariance_factors: vec![vec![0.]],
            loadings: vec![vec![0.]],
            idiosyncratic_variances: vec![0.],

            // Given but yet again we don't care?
            mu_assets: mu_assets.to_vec(),
            covariance_assets: covariance_assets.to_vec(),

            // This is what is actually used for sampling
            periods_to_sample,
            normal_distribution,
            rng,
        })
    }

    pub fn normal(
        means: &[f64],
        cov: &[f64],
        periods_to_sample: usize,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let rng = if let Some(s) = seed {
            ChaCha20Rng::seed_from_u64(s)
        } else {
            ChaCha20Rng::from_entropy()
        };

        let normal_distribution = MultivariateNormal::new(means.to_vec(), cov.to_vec()).map_err(|e| format!("Failed to create distribution for normal sampler: {}", e))?;
        Ok(Sampler::Normal {
            periods_to_sample,
            normal_distribution,
            rng,
        })
    }

    fn generate_covariance_matrix(number_of_factors: usize) -> Result<Vec<Vec<f64>>, String> {
        if number_of_factors == 0 {
            return Err("Number of factors must be greater than zero".to_string());
        }
        let mut rng = ChaCha20Rng::from_entropy();

        let uniform = Uniform::new(0.01, 0.2);

        let mut m = vec![vec![0.0; number_of_factors]; number_of_factors];
        for i in 0..number_of_factors {
            m[i][i] = rng.sample(uniform);
        }
        Ok(m)
    }

    fn compute_asset_covariance(
        loadings: &Vec<Vec<f64>>,
        covariance_factors: &Vec<Vec<f64>>,
        idiosyncratic_variances: &Vec<f64>,
    ) -> Result<Vec<Vec<f64>>, String> {
        let assets = loadings.len();
        let factors = covariance_factors.len();
        if loadings[0].len() != factors {
            return Err("Incompatible dimensions".to_string());
        }
        if idiosyncratic_variances.len() != assets {
            return Err("Incompatible dimensions".to_string());
        }

        let mut bsf = vec![vec![0.0; factors]; assets];
        for i in 0..assets {
            for j in 0..factors {
                for k in 0..factors {
                    bsf[i][j] += loadings[i][k] * covariance_factors[k][j];
                }
            }
        }

        let mut cov = vec![vec![0.0; assets]; assets];
        for i in 0..assets {
            for j in 0..assets {
                for k in 0..factors {
                    cov[i][j] += bsf[i][k] * loadings[j][k];
                }
            }
        }

        for i in 0..assets {
            cov[i][i] += idiosyncratic_variances[i];
        }
        Ok(cov)
    }

    pub fn sample_returns(&mut self) -> Vec<Vec<f64>> {
        match self {
            Sampler::FactorModel {
                normal_distribution,
                periods_to_sample,
                rng,
                ..
            } => normal_distribution
                .clone()
                .sample_iter(rng)
                .take(*periods_to_sample)
                .map(|row| row.as_slice().to_vec())
                .collect(),

            Sampler::Normal {
                normal_distribution,
                periods_to_sample,
                rng,
            } => normal_distribution
                .clone()
                .sample_iter(rng)
                .take(*periods_to_sample)
                .map(|row| row.as_slice().to_vec())
                .collect(),

            Sampler::SeriesGAN(periods_to_sample) => vec![vec![1.0]; *periods_to_sample],
        }
    }

    pub fn reseed(&mut self, seed: u64) {
        match self {
            Sampler::FactorModel { rng, .. } | Sampler::Normal { rng, .. } => {
                *rng = ChaCha20Rng::seed_from_u64(seed);
            }
            _ => {}
        }
    }

    pub fn find_min_max(raw_sequence: &[Vec<f64>]) -> Result<(Vec<(f64, f64)>, usize), String> {
        if raw_sequence.is_empty() {
            return Err("Passed an empty sequence".to_string());
        }
        let dim = raw_sequence[0].len();
        let mut mm = vec![(f64::INFINITY, f64::NEG_INFINITY); dim];
        for row in raw_sequence {
            for (i, &v) in row.iter().enumerate() {
                let (min, max) = &mut mm[i];
                *min = (*min).min(v);
                *max = (*max).max(v);
            }
        }
        Ok((mm, dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_covariance_matrix_valid_size() {
        let number_of_factors = 3;
        let result = Sampler::generate_covariance_matrix(number_of_factors);
        assert!(result.is_ok());
        let cov = result.unwrap();
        assert_eq!(cov.len(), number_of_factors);
        for row in cov {
            assert_eq!(row.len(), number_of_factors);
        }
    }

    #[test]
    fn test_generate_covariance_matrix_zero() {
        assert!(Sampler::generate_covariance_matrix(0).is_err());
    }
}
