//! SuperBrain 2.0 Rust 内核 PyO3 绑定。
//! 暴露：SIMD 点积 + HNSW 向量索引。

use pyo3::prelude::*;

mod dot;
mod hnsw;

use hnsw::Hnsw;

/// SIMD 加速的向量点积（AVX2 + 标量回退）。
#[pyfunction]
fn simd_dot(a: Vec<f32>, b: Vec<f32>) -> f32 {
    dot::dot(&a, &b)
}

/// HNSW 层次图向量索引：add 向量，search 返回 top-k (id, 点积)。
#[pyclass]
struct HnswIndex {
    inner: Hnsw,
}

#[pymethods]
impl HnswIndex {
    #[new]
    #[pyo3(signature = (dim, m=16, ef=32, seed=42))]
    fn new(dim: usize, m: usize, ef: usize, seed: u64) -> Self {
        HnswIndex { inner: Hnsw::new(dim, m, ef, seed) }
    }

    /// 插入一个向量，返回其 id。
    fn add(&mut self, vec: Vec<f32>) -> usize {
        self.inner.add(vec)
    }

    /// 查询与 query 最相似的 top-k，返回 [(id, 点积), ...]（按点积降序）。
    #[pyo3(signature = (query, k=5, ef=32))]
    fn search(&self, query: Vec<f32>, k: usize, ef: usize) -> Vec<(usize, f32)> {
        self.inner.search(&query, k.max(1), ef.max(1))
    }

    /// 已插入的向量数量。
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// 支持 Python 内置 len()。
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Python 模块入口。
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simd_dot, m)?)?;
    m.add_class::<HnswIndex>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}