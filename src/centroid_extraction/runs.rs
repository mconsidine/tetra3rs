//! Run-length connected-region core shared by both extraction paths.
//!
//! A single raster sweep turns an arbitrary per-pixel `lit` predicate into
//! horizontal runs, merging 8-connected runs across rows with union-find.
//! Region payloads live with the callers: the fast path computes moments
//! from the run lists in a post-pass (lit pixels are ≪1% of the image, so
//! the second touch is nearly free), while the quality path runs its
//! annulus/moment/deblend stages per region. This replaces the generic
//! mask → labels connected-component labeling, which materialized a u8 mask
//! and a u32 labels buffer (~10 MB at 2 Mpix) and required a per-pixel label
//! test in every downstream stage.

/// A horizontal run of lit pixels: columns `c0..=c1` of `row`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Run {
    pub row: u32,
    pub c0: u32,
    pub c1: u32,
}

impl Run {
    #[inline]
    pub(super) fn len(&self) -> usize {
        (self.c1 - self.c0 + 1) as usize
    }
}

/// Result of [`sweep_runs`]: the runs in creation (row-major) order, each
/// run's dense region id, and the region count. Region ids are assigned in
/// order of first appearance, so iterating regions is deterministic.
pub(super) struct RunRegions {
    pub runs: Vec<Run>,
    pub region_of_run: Vec<u32>,
    pub n_regions: usize,
}

impl RunRegions {
    /// Group runs by region: `order[offsets[k] as usize..offsets[k + 1] as
    /// usize]` are the (ascending, hence row-major) run indices of region
    /// `k`. Counting sort — O(runs + regions).
    pub(super) fn group_by_region(&self) -> (Vec<u32>, Vec<u32>) {
        let mut counts = vec![0u32; self.n_regions + 1];
        for &reg in &self.region_of_run {
            counts[reg as usize + 1] += 1;
        }
        for k in 1..counts.len() {
            counts[k] += counts[k - 1];
        }
        let offsets = counts.clone();
        let mut cursor = counts;
        let mut order = vec![0u32; self.runs.len()];
        for (i, &reg) in self.region_of_run.iter().enumerate() {
            order[cursor[reg as usize] as usize] = i as u32;
            cursor[reg as usize] += 1;
        }
        (offsets, order)
    }
}

/// Run-length 8-connected labeling over `lit(row, col)`.
///
/// The predicate is called exactly once per pixel in raster order, so callers
/// may keep per-row state inside the closure (e.g. a row-blended background).
pub(super) fn sweep_runs(
    w: usize,
    h: usize,
    mut lit: impl FnMut(usize, usize) -> bool,
) -> RunRegions {
    // One provisional union-find label per run; label id == run index.
    let mut parents: Vec<u32> = Vec::new();
    let mut runs: Vec<Run> = Vec::new();
    // Runs of the previous / current row: (c0, c1, label).
    let mut prev: Vec<(u32, u32, u32)> = Vec::new();
    let mut cur: Vec<(u32, u32, u32)> = Vec::new();

    for r in 0..h {
        cur.clear();
        let mut start: Option<usize> = None;
        for c in 0..w {
            if lit(r, c) {
                if start.is_none() {
                    start = Some(c);
                }
            } else if let Some(s) = start.take() {
                let label = runs.len() as u32;
                parents.push(label);
                runs.push(Run {
                    row: r as u32,
                    c0: s as u32,
                    c1: (c - 1) as u32,
                });
                cur.push((s as u32, (c - 1) as u32, label));
            }
        }
        if let Some(s) = start.take() {
            let label = runs.len() as u32;
            parents.push(label);
            runs.push(Run {
                row: r as u32,
                c0: s as u32,
                c1: (w - 1) as u32,
            });
            cur.push((s as u32, (w - 1) as u32, label));
        }

        // Merge current-row runs with 8-connected previous-row runs. Both
        // lists are column-sorted, so a two-pointer sweep finds all overlaps.
        let (mut i, mut j) = (0usize, 0usize);
        while i < cur.len() && j < prev.len() {
            let (cs, ce, cl) = cur[i];
            let (ps, pe, pl) = prev[j];
            if ce + 1 < ps {
                i += 1; // current run ends left of (and not adjacent to) prev
            } else if pe + 1 < cs {
                j += 1; // prev run ends left of current
            } else {
                union(&mut parents, cl, pl);
                if ce < pe {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }

        std::mem::swap(&mut prev, &mut cur);
    }

    // Resolve roots and compact them to dense region ids in first-appearance
    // (row-major) order.
    let mut region_of_run = vec![0u32; runs.len()];
    let mut id_of_root = vec![u32::MAX; parents.len()];
    let mut n_regions = 0usize;
    for (i, reg) in region_of_run.iter_mut().enumerate() {
        let root = find(&mut parents, i as u32) as usize;
        if id_of_root[root] == u32::MAX {
            id_of_root[root] = n_regions as u32;
            n_regions += 1;
        }
        *reg = id_of_root[root];
    }

    RunRegions {
        runs,
        region_of_run,
        n_regions,
    }
}

/// Union-find root with path halving.
fn find(parents: &mut [u32], mut x: u32) -> u32 {
    while parents[x as usize] != x {
        let p = parents[x as usize];
        parents[x as usize] = parents[p as usize]; // halve
        x = parents[x as usize];
    }
    x
}

/// Link the roots of two labels.
fn union(parents: &mut [u32], a: u32, b: u32) {
    let ra = find(parents, a);
    let rb = find(parents, b);
    if ra != rb {
        parents[ra as usize] = rb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regions_of(mask: &[&[u8]]) -> RunRegions {
        let h = mask.len();
        let w = mask[0].len();
        sweep_runs(w, h, |r, c| mask[r][c] != 0)
    }

    #[test]
    fn test_sweep_runs_basic() {
        let rr = regions_of(&[
            &[1, 1, 0, 0, 1],
            &[0, 1, 0, 1, 0], // diagonal touch joins col-4/row-0 with col-3/row-1
            &[0, 0, 0, 0, 0],
            &[1, 0, 0, 0, 1],
        ]);
        assert_eq!(rr.n_regions, 4);
        let (offsets, order) = rr.group_by_region();
        // Region of the first run (top-left pair) contains 2 runs.
        let first = rr.region_of_run[0] as usize;
        assert_eq!((offsets[first + 1] - offsets[first]), 2);
        // Total pixels across all runs.
        let total: usize = order.iter().map(|&i| rr.runs[i as usize].len()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn test_sweep_runs_full_row_and_empty() {
        let rr = regions_of(&[&[1, 1, 1], &[1, 1, 1]]);
        assert_eq!(rr.n_regions, 1);
        assert_eq!(rr.runs.len(), 2);

        let rr = regions_of(&[&[0, 0], &[0, 0]]);
        assert_eq!(rr.n_regions, 0);
        assert!(rr.runs.is_empty());
    }
}
