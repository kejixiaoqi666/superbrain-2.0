//! SIMD 向量点积：AVX2 加速（f32x8 FMA），标量回退。
//! 用于向量相似度计算，是检索热点的核心。

/// 标量点积（正确实现，回退路径）
#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// AVX2 加速点积（需要 CPU 支持 AVX2 + FMA）
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    // 主循环：每次 8 个 f32（256-bit）
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += 8;
    }
    // 水平求和：高 128 + 低 128
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    let mut sum = _mm_cvtss_f32(s);
    // 尾部标量
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// 归一化向量的点积（余弦等价的快速路径）
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: 已在运行时检测到 avx2+fma
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dot_matches_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let s = dot_scalar(&a, &b);
        let d = dot(&a, &b);
        assert!((s - d).abs() < 1e-4, "scalar={} avx={}", s, d);
    }
}
