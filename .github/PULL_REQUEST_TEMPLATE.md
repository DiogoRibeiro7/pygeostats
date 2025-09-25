# Pull Request

## Summary 

<!-- Short, clear summary of the change. What does this PR do? -->

 ## Motivation & Context 

<!-- Why is this change needed? Which problem does it solve? --> - Closes #

<issue-id> (if applicable)

## Type of Change
- [ ] feat: New feature
- [ ] fix: Bug fix
- [ ] perf: Performance improvement
- [ ] docs: Documentation only
- [ ] refactor: Code restructure without behavior change
- [ ] test: Adding or fixing tests
- [ ] build/chore: Build system or CI changes
- [ ] style: Formatting (no code change)
- [ ] breaking: Breaking change (see section below)

## Breaking Changes
<_comment> Describe API breaks or behavior changes. Migration notes if any. </_comment><h2 id="how-has-this-been-tested-">How Has This Been Tested?</h2><_comment> Be specific. List commands, datasets/shapes, and platforms. </_comment>
- [ ] Unit tests
- [ ] Property-based tests (hypothesis) where relevant
- [ ] Manual checks
- [ ] Cross-platform (Linux/macOS/Windows) when applicable

## Benchmarks (core numerics only)
<_comment> Paste key results from benchmarks/ scripts if you touched variogram/kriging kernels or solvers. </_comment>
- CPU / OS:
- Python / Rust / BLAS:
- Scripts run:
  - <code>python benchmarks/benchmark_variogram.py</code>
  - <code>python benchmarks/compare_with_gstat.py</code>
- Results summary:

## Documentation
- [ ] Docstrings updated
- [ ] Sphinx builds locally (<code>make -C docs html</code>)
- [ ] Examples/notebooks updated if behavior changed

## Screenshots / Plots
<_comment> Optional: figures that help reviewers understand the change. </_comment><h2 id="checklist">Checklist</h2><ul>
  <li>[ ] Code is typed (public APIs) and documented</li>
  <li>[ ] <code>black</code> / <code>ruff</code> pass locally</li>
  <li>[ ] <code>mypy</code> passes for <code>src/python/pyspatialstats/</code></li>
  <li>[ ] <code>cargo fmt</code> and <code>cargo clippy -D warnings</code> pass</li>
  <li>[ ] Tests added/updated and <code>pytest -v</code> is green</li>
  <li>[ ] Coverage not reduced meaningfully</li>
  <li>[ ] No large files added; small synthetic data only</li>
  <li>[ ] Changelog entry (if maintaining CHANGELOG)</li>
  <li>[ ] I have read and followed <code>CONTRIBUTING.md</code></li>
</ul><h2 id="additional-notes">Additional Notes</h2><_comment> Anything else reviewers should know. </_comment></issue-id>
