# Degree-1/2 Elimination Implementation Notes

This note records follow-up review comments for the exact degree-1/2 elimination
path implemented in [`src/elimination.jl`](../src/elimination.jl). The current
implementation is substantially more optimized than the earlier dictionary-based
approach: it reads the base graph from the CSC matrix, uses lazy deletion via
`alive`, stores fill edges in per-node spill vectors, deduplicates with an
epoch-stamped sparse accumulator, and rebuilds the reduced matrix with a dense
survivor index map.

Overall, this is a good optimization direction and should be much faster for the
intended near-tree workloads. The remaining comments below are primarily about
making the complexity claims more precise, adding benchmark coverage, and
reducing residual allocation costs.

## 1. Complexity comment may overstate amortized scanning behavior

The implementation comment currently says that compaction means every adjacency
cell is scanned `O(1)` times overall. That is stronger than what the current code
strictly guarantees.

`compact_adjacency!` replaces a node's spill vectors with the deduplicated live
adjacency, but later fill edges can be appended to those same vectors by
`push_fill!`. If the node is compacted again before it is eliminated, the full
current spill vector can be rescanned, not just the newly appended entries.

Suggested follow-up:

- Reword the comment above `eliminate_deg12` to say that the implementation
  avoids per-node hash maps and consumes base CSC slices lazily, while spill
  entries may be rescanned when a node is compacted multiple times.
- Keep the important correctness statement: stale candidate nodes are compacted
  and exact-degree checked before elimination.
- Consider adding instrumentation or benchmarks that count repeated compactions
  and scanned spill entries.

## 2. Stale candidate compactions can be expensive on adversarial cores

The `deg` array is used as a lower-bound candidate heuristic. It is decremented
when live neighbors are eliminated, but it is not incremented when fill edges are
created. This is correct as a candidate mechanism because each popped node is
compacted and exact-degree checked before elimination.

However, when fill edges make a node's true degree much larger than the lower
bound, the node can be pushed as a candidate even though its exact degree is well
above two. The code will skip that node after compaction, but the compaction can
be expensive if the node has accumulated many spill entries.

Suggested follow-up:

- Add benchmark cases that stress stale candidate compactions, for example:
  - long low-degree chains attached to a high-degree core,
  - graphs where fill edges repeatedly touch the same core nodes,
  - near-tree graphs with enough off-tree edges to create a nontrivial core.
- Track benchmark counters such as number of calls to `compact_adjacency!`, stale
  skips with exact degree greater than two, scanned base entries, and scanned
  spill entries.
- If stale compaction dominates, consider tracking newly appended fill entries
  separately from already compacted adjacency, or maintaining a more informative
  candidate heuristic.

## 3. Reduced-matrix rebuild can cheaply preallocate triplets

The reduced matrix rebuild now uses a dense `pos` vector instead of a dictionary,
which is a good improvement. The triplet vectors `I`, `J`, and `V` are still
created empty and grown with repeated `push!` calls.

Suggested follow-up:

- Estimate the number of triplets before filling the reduced matrix.
- Call `sizehint!` on `I`, `J`, and `V` using that estimate.
- Benchmark this on cases where the surviving core is large, such as grid-like
  SDD matrices.

## 4. Preconditioner application still allocates on every call

The build-time elimination path is now more optimized, but the preconditioner
application path still allocates per call. In particular, `forward_elim` copies
the right-hand side, the closure creates a full-size zero solution vector, and
the reduced right-hand side is materialized from `y[EH.ind]`.

Suggested follow-up:

- Add an internal workspace for the eliminated preconditioner closure containing:
  - a full-size forward-substitution buffer,
  - a full-size solution buffer,
  - a reduced right-hand-side buffer,
  - optionally a reduced solution buffer.
- Add an in-place `forward_elim!` helper that copies the input into a reusable
  buffer and applies the forward elimination updates there.
- Replace slicing-based reduced RHS construction with an explicit gather into a
  preallocated buffer.
- Decide whether the returned solution must be copied before returning. If the
  same buffer is reused between calls, document the non-reentrant behavior
  clearly.

## 5. Clarify validation expectations for `eliminate = true`

The elimination branch runs `eliminate_deg12` before validating the reduced
matrix with `validateInput!`. This means the elimination routine assumes that the
original input already has Laplacian/SDD-compatible structure, especially
non-positive off-diagonal entries that become positive edge weights after
negation.

Suggested follow-up:

- Decide whether `eliminate_deg12` requires prevalidated input.
- If it does, document that precondition in the function docstring.
- If it does not, validate the original matrix before elimination or add
  lightweight validation checks to the elimination routine.
- Try to keep invalid-input errors consistent with the non-elimination path.

## Summary

The current array/spill/epoch implementation is a meaningful optimization over a
hash-map-based implementation and is likely appropriate for near-tree workloads.
The main remaining work is to verify performance with benchmarks, make the
amortized-complexity documentation more precise, preallocate reduced-matrix
triplets, and reduce allocations when the eliminated preconditioner is applied
repeatedly.
