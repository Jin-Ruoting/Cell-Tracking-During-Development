# Kaggle compute fallback for E038

Kaggle GPU compute may run the frozen development experiment when the primary
server is unavailable. Source changes still originate in the local repository,
are reviewed, committed and pushed before execution. The server workflow and
its strict replay checks are retained.

## Frozen comparison

- Control: E029, with the original seven geometric-fusion overrides.
- Candidate: [E038 observed one-frame bridges](POINT_GAP_EXPERIMENT_20260923.md).
  No E031 flow or E035 motion-EMA component is added. Their verified public
  scores are both 0.946, below E029's 0.947.
- Reconstruct the historical 64-movie selection using the rule from commit
  `394154a`, then require the existing corpus SHA256
  `276c09d16cddaf2e865896ce147161a7beb5a62142bf47c1f1bd7648f7643e7f`.
  Label metadata is read during this reconstruction. Inference receives only
  image links; no labels are passed to either model or the bridge policy.
- Smoke movies are `44b6_eb2880fc` and `6bba_969618f6`. They establish runtime,
  export and scoring compatibility only; their gains cannot select parameters.
- Require the historical E029 CSV SHA256 and official control score before
  accepting migration. If CSV parity fails, retain the diagnostic control
  score and stop before E038. Do not loosen the guard based on the result.
- Full advancement still requires score gain >=0.001, nondecreasing adjusted
  edge Jaccard, positive gains on both embryos and both alternating halves,
  more affected wins than losses, and positive affected median. Training
  overlap is unresolved; these remain development results, not independent CV.

## Private run package

`kaggle/build_kaggle_development.py` packages committed first-party modules,
the checksum-pinned external Notebook and a minimal official scoring runtime.
It executes no models and uploads nothing. The downloaded sources, generated
Notebook, manifests, weights, outputs and credentials must stay out of Git.
Runtime packages are installed from the mounted support pack, with Notebook
Internet access disabled. Separate processes isolate reference inference,
v5 peak capture, bridge construction and official scoring.

The submitted E029 v1 predictor is pinned after normalizing exactly its two
diagnostic log-directory literals to `/kaggle/working`. No algorithm expression
is normalized or ignored. This reproduces the historical server checksum when
those same literals are set to its old log directory. The guard runs before
new inference, and both raw and canonical hashes are retained.

If control inference and postprocessing complete but a migration audit fails,
`--completed-control-dir` can package the downloaded smoke control CSV,
predictor and original run/cohort receipts. This limited recovery verifies
provenance, predictor identity and file hashes, then repeats boundary/topology,
historical CSV and official-score checks on Kaggle. It does not rerun E029
inference or relax any E038 advancement gate. Reuse is explicit in the receipt.

External inputs:

- [Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion),
  Notebook SHA256 `f82a606e2e3289f1d6dce078d381dd7b0caf148c92bb25f2f05bea80d4e79a37`.
  Only its first six cells execute; the parameter sweep and validator do not.
- [Official scorer at 075fc5f](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f),
  codeload archive SHA256 `0e31329953304331d83f32b5d6ab40f1424ed6391ea2b4e6ad5b56eba62ab367`.
  The original BSD license and official scoring sources are retained.
- The three original `pilkwang` checkpoint/support datasets, plus
  [hengck23's reviewed v5 detector](https://www.kaggle.com/datasets/hengck23/hengck23-cell-point-detector-demo).
  All model and source checksums are checked before execution.

Example after committing and pushing local source:

```bash
python kaggle/build_kaggle_development.py \
  --owner YOUR_KAGGLE_USERNAME \
  --reference-notebook /tmp/biohub-geometric-fusion.ipynb \
  --scorer-archive /tmp/biohub-official-scorer-075fc5f.tar.gz \
  --mode smoke --output-dir /tmp/biohub-e038-smoke
kaggle kernels push -p /tmp/biohub-e038-smoke \
  --accelerator NvidiaTeslaT4 --timeout 7200
```

The metadata requires `is_private: true`. Verify the saved version and
visibility after upload. This is a development Notebook using training
movies; the launcher never calls competition submission APIs. Logs and
receipts are under `/kaggle/working/logs/e038-development`. The actual run
status, timing and results must be recorded in the private operational MD.

The builder allows `--mode full` only with `--smoke-receipt` pointing to a
successful downloaded `development_receipt.json` from the same frozen
protocol. Check remaining GPU quota and measured smoke timing before launching
the full run. Any promoted method still requires the existing author-selection
exclusion review, actual test-output audit and separately verified public score.
