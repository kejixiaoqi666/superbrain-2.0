//! HNSW 层次图向量索引（Malkov & Yashunin）。
//! 用点积作为相似度（归一化向量时 = 余弦），查询最大点积的 top-k。
//! 零依赖：内置 xorshift RNG 用于层级分配。

use std::cmp::Ordering;
use std::collections::HashSet;
use crate::dot::dot;

const MAX_LEVEL: usize = 32;

pub struct Hnsw {
    dim: usize,
    m: usize,                     // 每层最大连接数
    ef: usize,                    // 插入时候选集大小
    vectors: Vec<Vec<f32>>,
    levels: Vec<usize>,           // 每个节点最高层
    graph: Vec<Vec<Vec<usize>>>,  // graph[id][level] -> 邻居 id
    entry: usize,
    max_level: usize,
    level_mult: f64,
    rng_state: u64,
    has_entry: bool,
}

impl Hnsw {
    pub fn new(dim: usize, m: usize, ef: usize, seed: u64) -> Self {
        let m = m.max(2);
        Hnsw {
            dim,
            m,
            ef: ef.max(m),
            vectors: Vec::new(),
            levels: Vec::new(),
            graph: Vec::new(),
            entry: 0,
            max_level: 0,
            level_mult: 1.0 / (m as f64).ln(),
            rng_state: seed.max(1),
            has_entry: false,
        }
    }

    #[inline]
    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn rand_level(&mut self) -> usize {
        let r = (self.next_rand() >> 11) as f64 / (1u64 << 53) as f64;
        let u = (r + 1e-12).min(1.0 - 1e-12);
        ((-u.ln()) * self.level_mult).floor() as usize
    }

    fn score(&self, a: &[f32], b_id: usize) -> f32 {
        dot(a, &self.vectors[b_id])
    }

    fn worse<'a>(a: &'a (f32, usize), b: &'a (f32, usize)) -> Ordering {
        // 越小越"差"（点积最小 = 最差）
        a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
    }

    /// 在指定层贪心搜索，返回 ef 候选集（点积降序）
    fn search_layer(&self, q: &[f32], entry_points: &[usize], ef: usize, level: usize) -> Vec<(f32, usize)> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: Vec<(f32, usize)> = Vec::new();  // 待扩展
        let mut results: Vec<(f32, usize)> = Vec::new();     // top-ef（降序，last=最差）
        for &ep in entry_points {
            if visited.insert(ep) {
                let s = self.score(q, ep);
                candidates.push((s, ep));
                results.push((s, ep));
            }
        }
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        while !candidates.is_empty() {
            // 取最好候选
            let bi = candidates.iter().enumerate()
                .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap_or(Ordering::Equal)).unwrap().0;
            let (best_s, best_id) = candidates.swap_remove(bi);
            // 最好候选都不如 results 最差 且已满 → 停止
            if results.len() >= ef && best_s < results.last().unwrap().0 { break; }
            if level < self.graph[best_id].len() {
                for &nbr in self.graph[best_id][level].iter() {
                    if !visited.insert(nbr) { continue; }
                    let s = self.score(q, nbr);
                    if results.len() < ef || s > results.last().unwrap().0 {
                        candidates.push((s, nbr));
                        results.push((s, nbr));
                        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }
        results
    }

    pub fn add(&mut self, vec: Vec<f32>) -> usize {
        let id = self.vectors.len();
        self.vectors.push(vec);
        let level = self.rand_level();
        self.levels.push(level);
        self.graph.push(vec![Vec::new(); level + 1]);
        if !self.has_entry {
            self.has_entry = true;
            self.entry = id;
            self.max_level = level;
            return id;
        }
        let mut cur_entries = vec![self.entry];
        // 高层只搜索找入口，不建连接
        for lc in ((level + 1)..=self.max_level.max(level)).rev() {
            cur_entries = self.search_layer(&self.vectors[id], &cur_entries, 1, lc)
                .into_iter().take(1).map(|x| x.1).collect();
        }
        // 建连接层
        for lc in (0..=level).rev() {
            let found = self.search_layer(&self.vectors[id], &cur_entries, self.ef, lc);
            let nbrs: Vec<usize> = found.iter().map(|x| x.1).take(self.m).collect();
            for &nbr in &nbrs {
                self.graph[id][lc].push(nbr);
                if lc < self.graph[nbr].len() {  // 邻居需有此层，防越界
                    self.graph[nbr][lc].push(id);
                    if self.graph[nbr][lc].len() > self.m {
                        self.prune_neighbors(nbr, lc, id);
                    }
                }
            }
            cur_entries = nbrs;
        }
        if level > self.max_level {
            self.max_level = level;
            self.entry = id;
        }
        id
    }

    /// 连接过多时按与新节点相似度截断
    fn prune_neighbors(&mut self, node: usize, level: usize, new_id: usize) {
        let v = self.vectors[new_id].clone();
        // 预算好所有邻居与新节点的分数（避免在排序闭包里借用 self）
        let mut scored: Vec<(f32, usize)> = self.graph[node][level]
            .iter().map(|&n| (dot(&v, &self.vectors[n]), n)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)); // 降序
        self.graph[node][level] = scored.into_iter().take(self.m).map(|x| x.1).collect();
    }

    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if !self.has_entry { return Vec::new(); }
        let mut cur_entries = vec![self.entry];
        // 高层下降：保留多个入口（避免只取 best 1 个导致丢失到目标的路径）
        for lc in (1..=self.max_level).rev() {
            let r = self.search_layer(q, &cur_entries, self.ef.min(8).max(1), lc);
            cur_entries = r.into_iter().take(8).map(|x| x.1).collect();
        }
        // 层 0 起点保底：高层下降结果 + 入口点（避免高层引导带偏丢失可达目标）
        if !cur_entries.contains(&self.entry) {
            cur_entries.push(self.entry);
        }
        let mut found = self.search_layer(q, &cur_entries, ef.max(cur_entries.len()), 0);
        found.sort_by(|a, b| Self::worse(b, a));  // 点积降序（worse 本身是升序，反转取最好）
        found.truncate(k);
        found.into_iter().map(|(s, id)| (id, s)).collect()
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hnsw_finds_nn() {
        let mut h = Hnsw::new(8, 8, 16, 42);
        // 插入 100 个随机单位向量
        let mut s: u64 = 7;
        for _ in 0..100 {
            let mut v = vec![0f32; 8];
            for c in 0..8 {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                v[c] = ((s >> 33) as f32 / (1u32 << 31) as f32) - 1.0;
            }
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in v.iter_mut() { *x /= norm; }
            h.add(v);
        }
        // 查询一个已插入向量（应命中自身 top1）
        let q = h.vectors[50].clone();
        let r = h.search(&q, 5, 16);
        assert_eq!(r[0].0, 50, "HNSW 应找到自身, 实际 top1={:?}", r[0]);
    }
}
