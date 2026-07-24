# Leaderboard

## Snapshot: 2026-07-24 12:42 CST

- Kaggle leaderboard export time: `2026-07-24T04:42:42Z`.
- Competition teams: 1,579.
- Strict top-10% target rank: 157 or better
  (`floor(1,579 * 0.10)`).
- Score at rank 157: `0.908`.
- Current leader score: `0.931`.
- Team `Steven #2`, member `buaaauto`: rank 102 with score `0.908`.
- Exactly three teams are strictly above `0.92`:
  `TWEAK` at `0.931`, `Amin` at `0.929`, and `chrome frames` at `0.923`.
- Current `> 0.92` objective: not yet achieved; displayed gap to `0.921` is
  `+0.013`.

## Snapshot: 2026-07-24 01:00 CST

- Kaggle leaderboard export time: `2026-07-23T17:00:39Z`.
- Competition teams: 1,566.
- Strict top-10% target rank: 156 or better
  (`floor(1,566 * 0.10)`).
- Score at rank 156: `0.904`.
- Current leader score: `0.929`.
- Team `Steven #2`, member `buaaauto`: rank 78 with score `0.908`.
- Rank fraction: `78 / 1,566 = 0.049808...`, approximately top 4.98%.
- Result: the strict top-10% target is achieved with a 78-rank margin to the
  cutoff.

## Snapshot: 2026-07-23 17:05 CST

- Competition teams: 1,549.
- Strict top-10% target rank: 154 or better
  (`floor(1,549 * 0.10)`).
- Score at displayed position 154: `0.902`.
- Current leader score: `0.929`.
- Rescore after the division-metric patch was reported complete on 2026-07-23.
- Submission `54923913` was pending at the time of this snapshot.

The target is evaluated from the live Kaggle leaderboard at each submission
check; the cutoff is expected to move.

Earlier planning used rank 155 by rounding 10% upward. This record uses the
stricter mathematical requirement `rank / team_count <= 0.10`, so rank 154 is
the current maximum qualifying rank.

## Our Submissions

| Experiment | Submission ID | Public score | Rank / teams | Percentile | Status |
|---|---:|---:|---:|---:|---|
| E000 | 54923913 | `0.908` | 102 / 1,579 | top 6.46% | Complete; earlier top-10% target achieved, current `> 0.92` target open |

## Held Candidates

| Experiment | Kaggle version | Fixed-8 score | Decision |
|---|---:|---:|---|
| E001 no-rescue control | 3 | `0.8789316335` | Do not submit; local delta over E000 is negligible and E000 achieved the target |
| E003 zero-motion LAP | local | best `0.8680921294` on disjoint holdouts | Reject; tied E001 at best |
| Public dual-seed router | 33 | Public Score `0.909` | Do not submit unchanged; inspected legal approach is below target |
