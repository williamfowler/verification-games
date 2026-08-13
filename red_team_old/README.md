# red_team_old — retired Phase II v1 artifacts

Superseded by the Phase II **v3** plan (deployment-framed, under-report-primary,
one-variable/GT-preserving). Kept for provenance and because their *results* are
real negatives the v3 findings build on — not deleted, just out of the active
trial budget.

## What's here and why it was retired

| file | what | why retired |
|---|---|---|
| `redteam_configs_v1.py` | v1 RED_CONFIGS: S1 precision (fp32), S2 mem_decoy, S2′ nvlink_decoy, S3 atypical (ff/seq/layer/d_model) | Decoy strategies over-report + blow budget (energy model isn't the surface on Volta). v1 S3 changed **GT-altering** knobs, violating the v3 one-variable/GT-preserving invariant. |
| `adversarial_workload_v1.py` | decoy-bearing entry point (`mem_decoy` + `DecoyTraffic`/`make_decoy_allreduce`) | `mem_decoy`/`nvlink_decoy` retired. The active `red_team/adversarial_workload.py` keeps only `none`/`split`/`throttle`. |
| `score_redteam_v1.py` | v1 scorer (over-report-magnitude framing) | Reworked in v3 to foreground **signed error** and **under-report success vs the benign noise band**. |
| `red_results_trial1_records.json`, `live_*.json`, `redteam_scores.json`, `red_results_trial1.txt` | v1 single-trial records + scores | v1 data; v3 recollects under the parent-linked, GT-preserving configs. |
| `fig_redteam_error.png`, `fig_redteam_efficiency.png` | v1 figures (all-over-report bars; efficiency bars) | Replaced by v3 signed-error-distribution + evasion-vs-budget figures. |
| `collect_a.log`, `group_b.log` | v1 collection logs | historical. |

## The v1 result that stands (context for v3)

Every offline "hide-FLOPs" attack (precision, DRAM decoy, NVLink decoy, atypical
shape) made the estimator **over-report** and exceeded the 2× budget — because on
Volta fp16 is simultaneously the baseline, the max-throughput precision, and the
calibration precision, so nothing lowers measured energy per true FLOP. Only the
live-daemon **workload split** evaded (session fragmentation). v3 keeps split,
keeps throttle as a documented gate-failure, and adds real under-report levers
(**batch-inflation**, **power-capping**) that attack the compute-energy term
instead of the decoy/precision channels. See `writeup/redteam_phase2_findings.md`
(marked v1) for the full write-up.
