# Repository review

## Completed in the publication cleanup

- Added ignore rules for environments, model weights, local results, credentials,
  machine files, and personal notes.
- Moved application-oriented notes/drafts into ignored `local-notes/`; replaced the
  report generator's personal draft with a neutral evidence summary.
- Removed the tracked Finder metadata file from the working tree.
- Updated stale execution-status documentation with real, failed-quality findings.
- Added contribution guidance, offline GitHub Actions CI, and package README/URLs.
- Added report quality diagnostics and corrected cache agreement so invalid parsed
  outputs do not count as valid answer matches.

The removed files remain in earlier commits. Ignore rules do not rewrite Git history.
The next commit must include the tracked deletions as well as new/modified files.
This cleanup does not publish, push, or change repository visibility.

## License and history

Source code now uses the MIT License. Model weights and dependencies have separate terms.
Ignored personal material still exists in earlier commits; this task does not rewrite history.

## Version 2 implementation update

The baseline/holdout protocol, unambiguous performance formats, native integration checks,
controlled cache-hit study, concurrent HTTP driver, source fingerprints and evidence export
are implemented. See [PROTOCOL.md](PROTOCOL.md). Native v2 results remain blocked by
Metal access in the execution sandbox; no passing model baseline is yet claimed.
The following original priorities remain useful research directions; implementation of
an experiment is distinct from validation of its results.

## Research and engineering priorities

1. Establish a useful baseline: current decision accuracy is 14/52. Investigate common
   prompt/model suitability on a development set; retain a new held-out evaluation.
2. Remove conflicting JSON/prose instructions in a versioned performance workload;
   add natural JSON response timing. Keep old evidence immutable and rerun all variants.
3. Add native GPU integration checks for cache copying, EOS behavior, memory pressure,
   and model conversion. Offline mocks cannot substitute for these tests.
4. Add mixed cache-hit workloads, eviction scenarios, concurrency/load sweeps, and
   server-side latency before making service-capacity claims.
5. Publish a sanitized raw evidence bundle plus dependency lock for independently
   auditable performance claims. Add source commit/dirty state to future run provenance;
   current metadata identifies models/workload/environment but not runner code revision.
6. Strengthen report ingestion for arbitrary external evidence (schema/version validation,
   complete expected quality-case coverage, contiguous rounds, finite values). Current
   checks target files produced by this runner, not untrusted third-party uploads.
7. Validate NVIDIA command generation against a pinned installed vLLM environment.
   It remains a proposed extension, not a tested backend.

Public experimental repositories can contain negative results. Preserve the failed gate
and scope limitations instead of presenting these timings as production readiness.
