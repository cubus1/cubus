//! Linear algebra helpers: Cholesky solve, condition number, resampling.

use crate::sweep::Base;

/// Cholesky decomposition and solve for symmetric positive definite matrix.
pub fn cholesky_solve(gram: &[f64], rhs: &[f64], k: usize) -> Vec<f64> {
    // Cholesky: G = L · L^T
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = gram[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                // Diagonal: handle near-zero for numerical stability
                l[i * k + j] = if sum > 1e-12 { sum.sqrt() } else { 1e-6 };
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }

    // Forward: L · y = rhs
    let mut y = vec![0.0f64; k];
    for i in 0..k {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[i * k + j] * y[j];
        }
        y[i] = sum / l[i * k + i];
    }

    // Backward: L^T · alpha = y
    let mut alpha = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for j in (i + 1)..k {
            sum -= l[j * k + i] * alpha[j];
        }
        alpha[i] = sum / l[i * k + i];
    }

    alpha
}

/// Condition number estimate: ratio of largest to smallest diagonal of Cholesky.
pub fn condition_number(gram: &[f64], k: usize) -> f64 {
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = gram[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                l[i * k + j] = if sum > 1e-12 { sum.sqrt() } else { 1e-6 };
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }

    let mut max_diag = 0.0f64;
    let mut min_diag = f64::MAX;
    for i in 0..k {
        let d = l[i * k + i];
        max_diag = max_diag.max(d);
        min_diag = min_diag.min(d);
    }

    if min_diag < 1e-12 {
        f64::MAX
    } else {
        (max_diag / min_diag).powi(2)
    }
}

/// Upsample i8 vector to f32 at higher dimensionality.
/// Each warm element maps to d_hot/d_warm hot elements via interpolation.
pub fn upsample_to_f32(v: &[i8], d_hot: usize) -> Vec<f32> {
    let d_warm = v.len();
    if d_warm == 0 {
        return vec![0.0f32; d_hot];
    }
    let ratio = d_hot as f32 / d_warm as f32;
    let mut result = vec![0.0f32; d_hot];

    for j in 0..d_hot {
        let warm_pos = j as f32 / ratio;
        let idx_low = (warm_pos.floor() as usize).min(d_warm - 1);
        let idx_high = (idx_low + 1).min(d_warm - 1);
        let frac = warm_pos - warm_pos.floor();
        result[j] = v[idx_low] as f32 * (1.0 - frac) + v[idx_high] as f32 * frac;
    }

    result
}

/// Downsample f32 vector to i8 at lower dimensionality.
/// Groups of (d_hot/d_warm) elements are averaged then quantized.
pub fn downsample_to_base(v: &[f32], d_warm: usize, base: Base) -> Vec<i8> {
    let d_hot = v.len();
    if d_warm == 0 || d_hot == 0 {
        return vec![0i8; d_warm];
    }
    let group = d_hot / d_warm;
    let group = group.max(1);
    let min = base.min_val() as f32;
    let max = base.max_val() as f32;

    let mut result = vec![0i8; d_warm];
    for i in 0..d_warm {
        let start = i * group;
        let end = (start + group).min(d_hot);
        let count = end - start;
        if count == 0 {
            continue;
        }
        let avg: f32 = v[start..end].iter().sum::<f32>() / count as f32;
        result[i] = avg.round().clamp(min, max) as i8;
    }
    result
}

/// Dot products: matrix × vector (templates × holograph).
pub fn dot_matrix_vector(templates: &[Vec<i8>], v: &[i8]) -> Vec<f64> {
    templates
        .iter()
        .map(|t| {
            t.iter()
                .zip(v.iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum()
        })
        .collect()
}

/// Gram matrix: templates × templates^T.
pub fn gram_matrix(templates: &[Vec<i8>]) -> Vec<f64> {
    let k = templates.len();
    let mut gram = vec![0.0f64; k * k];
    for i in 0..k {
        for j in i..k {
            let dot: f64 = templates[i]
                .iter()
                .zip(templates[j].iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            gram[i * k + j] = dot;
            gram[j * k + i] = dot;
        }
    }
    gram
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky_solve_identity() {
        // Identity gram: solution = rhs
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let rhs = vec![3.0, 7.0];
        let sol = cholesky_solve(&gram, &rhs, 2);
        assert!((sol[0] - 3.0).abs() < 1e-10);
        assert!((sol[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_solve_2x2() {
        // [[4, 2], [2, 3]] · x = [8, 7] → x = [1, 2] (check: 4+4=8, 2+6=8 — wait)
        // Actually: 4*1 + 2*2 = 8 ✓, 2*1 + 3*2 = 8 ≠ 7
        // Let's use: [[4, 2], [2, 3]] · x = [8, 8] → x = [1, 2]
        let gram = vec![4.0, 2.0, 2.0, 3.0];
        let rhs = vec![8.0, 8.0];
        let sol = cholesky_solve(&gram, &rhs, 2);
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_condition_number_identity() {
        let gram = vec![1.0, 0.0, 0.0, 1.0];
        let cond = condition_number(&gram, 2);
        assert!((cond - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_upsample_doubles_length() {
        let v = vec![1i8, -1, 2, 0];
        let up = upsample_to_f32(&v, 8);
        assert_eq!(up.len(), 8);
        // First and last should be close to original endpoints
        assert!((up[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_downsample_halves_length() {
        let v = vec![1.0f32, 1.0, -1.0, -1.0, 2.0, 2.0, 0.0, 0.0];
        let down = downsample_to_base(&v, 4, Base::Signed(5));
        assert_eq!(down.len(), 4);
        assert_eq!(down[0], 1); // avg of [1,1]
        assert_eq!(down[1], -1); // avg of [-1,-1]
        assert_eq!(down[2], 2); // avg of [2,2]
        assert_eq!(down[3], 0); // avg of [0,0]
    }

    #[test]
    fn test_upsample_downsample_round_trip() {
        let v = vec![1i8, -1, 2, 0, -2, 1];
        let up = upsample_to_f32(&v, 24);
        let down = downsample_to_base(&up, 6, Base::Signed(5));
        for i in 0..v.len() {
            assert!(
                (v[i] - down[i]).abs() <= 1,
                "round trip: {} vs {} at {}",
                v[i],
                down[i],
                i
            );
        }
    }

    #[test]
    fn test_gram_matrix_orthogonal() {
        // Two orthogonal vectors
        let t = vec![vec![1i8, 0, 1, 0], vec![0, 1, 0, 1]];
        let gram = gram_matrix(&t);
        assert_eq!(gram[0], 2.0); // [1,0,1,0] · [1,0,1,0]
        assert_eq!(gram[1], 0.0); // orthogonal
        assert_eq!(gram[3], 2.0); // [0,1,0,1] · [0,1,0,1]
    }

    #[test]
    fn test_dot_matrix_vector() {
        let t = vec![vec![1i8, 2, 3], vec![4, 5, 6]];
        let v = vec![1i8, 1, 1];
        let dots = dot_matrix_vector(&t, &v);
        assert_eq!(dots[0], 6.0); // 1+2+3
        assert_eq!(dots[1], 15.0); // 4+5+6
    }
}
