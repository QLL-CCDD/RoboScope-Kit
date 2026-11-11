# Current RollBall runtime diagnosis

**2026-11-11: current baseline reproduced; no new runtime gain is claimed.**
Measurements use the unmerged [PR #99 integration candidate](https://github.com/MINT-SJTU/RoboScope-Kit/pull/99); full issue #25 acceptance remains incomplete.
The frozen b52 source is `db473261e41c4663a7350bbb79816789fc632f48`. Both paths use the same
real FP32 checkpoint, task/controller, 44D state, seeds 0-99, 80-tick horizon and N10.
SDK uses fixed K1/G1, no autotune, MPS, profiler or trajectory hashing.
There are 10 vector groups, 800 control/RPC batches and 8,000 environment control steps.
100/20 Hz implies 4,000 shared rollout PhysX step calls, integrating 40,000 lane microsteps;
constructor/reset settling steps are outside that source-derived rollout count.

## Untraced baseline

| Metric | SDK | Direct native GPU-vector reference |
|---|---:|---:|
| Complete command wall s | 42.325058 | 39.911208 |
| Reported evaluation/formal wall s | 38.520170 | 38.495864 |
| Reported formal episodes/h | 9345.753 | 9351.654 |
| Reported formal control steps/s | 207.683 | 207.815 |
| Command episodes/h | 8505.600 | 9020.023 |
| Command control steps/s | 189.013 | 200.445 |
| Mean / peak GPU busy % | 24.722 / 64 | 25.844 / 71 |
| Mean / sampled peak VRAM GiB | 3.640 / 4.914 | 3.381 / 4.440 |
| Waited-child accounted mean CPU cores | 0.946 | 0.848 |
| Sampled mean / peak CPU cores | 0.880 / 1.113 | 0.840 / 1.039 |
| Success once / at end | 63/100 / 0/100 | 63/100 / 0/100 |
| Unsuccessful scored episodes | 37 | 37 |
| Worker failure / OOM | 0 / 0 | No process failure; no OOM observed |

Native `formal_wall_s` includes `imports_and_setup_s`; its narrower `evaluation_wall_s` is
37.022020 s. SDK `evaluation_s` starts after Policy/runtime setup and includes
Worker imports and queue polling. These distinct clocks are not a measured runtime speedup.
This native script directly uses upstream Gym, wrappers and original PPO weights. The original
author evaluator CLI (`run_upstream.py`) was **not** run in this new cohort.

250 ms NVML samples are device-wide arithmetic means and sampled peaks. SDK evaluation-only
mean busy is 26.935%; the common Worker
window is 27.893907 s with mean busy
33.541%. The SDK steady estimator
reports 258.121 steps/s after the first completed group;
completion-poll quantization and NVML sampling make this an approximate window.

Strict comparison passes 100 unique paired records: initial hashes, first actions, returns,
successes, seeds, lanes, step counts and actual masks match exactly. Per-step trajectory hashing
was disabled, so this is not a fresh 8,000-action trajectory-equality proof. All 100 episodes
normally truncate at the frozen horizon; a dedicated execution-timeout counter is unavailable.

## Non-overlapping SDK evaluation budget

| Component | Seconds | Evaluation % |
|---|---:|---:|
| environment_initialization_s | 10.928917 | 28.372 |
| environment_reset_s | 4.079078 | 10.589 |
| initial_observation_prepare_s | 0.002344 | 0.006 |
| policy_reset_s | 0.014931 | 0.039 |
| rollout_s | 19.268803 | 50.023 |
| policy_cleanup_s | 0.009906 | 0.026 |
| environment_close_s | 2.151255 | 5.585 |
| outside_group_spans_s | 2.064937 | 5.361 |

Policy load is 1.662925 s outside evaluation; runtime preparation is
0.603407 s. Outside-group residuals include uninstrumented Worker
startup, queues, result handling and polling; they are not pure import time.

| Nested rollout component | Seconds | Rollout % |
|---|---:|---:|
| step_with_observation_cpu_wall_s | 17.250680 | 89.526 |
| inference_wait_s | 1.769494 | 9.183 |
| observation_prepare_cpu_wall_s | 0.057333 | 0.298 |
| other_rollout_s | 0.191296 | 0.993 |

The first 10-lane group rollout took 2.221688 s. Across ten groups,
median/p95/min/max are 1.900818/2.097647/1.823056/2.221688 s;
the later nine groups have mean/median 1.894124/1.887174 s. Each group shares one
rollout wall clock among ten simultaneous episodes; these are ten samples, not 100 independent
episode-duration measurements. Init/reset/close optimizations are excluded from retained candidates.

## Separate profiler diagnostics

The built-in `--profile` N10 capture contains only a 0.736168 ms episode range and a 0.633211 ms
Policy reset. It has **no CUDA events**. Each sequential lane reset opens a new episode range,
closing the first before rollout. This run does not profile ManiSkill rollout. The independent
whole-tree capture successfully includes both Worker and Policy Server CUDA contexts.

The whole-tree run has 80 inference batches and 800 environment steps. Its 2.689751 s rollout
is a profiler diagnostic, not a formal speed measurement. Worker PID 74838 launches 185,142
kernels with 854.782472 ms summed kernel time; Policy PID 74815 launches 560 kernels with
1.046039 ms total. All CUDA kernel/copy/memset activity has 880.323426 ms union, over a
4.267042 s first-to-last-kernel interval that also includes initialization/reset. Do not add
overlapping kernel/API intervals to infer total application time.

The Worker main thread (PID/TID 74838) owns 255,984 CUDA API calls, totaling 1.808853 s.
`cudaLaunchKernel` takes 600.266 ms in 71,467 calls; `cuLaunchKernel` 403.172 ms in 113,675 calls.
`cuMemHostAlloc`/`cuMemFreeHost` consume 416.962/87.025 ms, including startup; they are not
established sustained allocations. Worker `cudaStreamSynchronize` totals 37.478 ms and
whole-tree stream synchronization APIs total 49.600 ms. This is a launch-heavy, serial path;
no single redundant device synchronization explains a large speedup. CPU API time overlaps
GPU execution and cannot be added to kernel time. OSRT poll/condition-wait sums span idle
threads/processes and are not the critical-path wait budget. CPU scheduling samples were disabled.

The three largest PhysX kernels sum to 493.029 ms: articulation internal constraints,
articulation stepping and tendon constraints. Reducing their solver iterations, contact rules or
physical steps would change the frozen evaluation and is excluded.

The unchanged actor has seven kernels per batch (four GEMMs, three activations). Mean summed
kernel execution is 0.013075 ms/batch. The first-to-last kernel span is 0.374540 ms mean;
input H2D to output D2H span is 1.477457 ms mean, or 0.523082 ms after the first inference.
These spans include CPU launch gaps. The first span includes 76.873051 ms of cold work.
They are not CUDA-event measurements from the formal baseline. Formal RPC latency remains
2.211867 ms/batch, including model execution, IPC and waiting; queue-only wait remains N/A.

State observations request no image capture; separate render CPU/GPU time was not measured.
The actual minimal Nsight Compute probe failed `ERR_NVGPUCTRPERM`; SM/Tensor/RT throughput
remains N/A. NVML busy and traced CUDA durations do not replace those counters.

## Sustained screening result

The CPU-dispatcher candidate was screened in ABBA order, 20 episodes per run,
seeds 0-19, same K1/G1/N10 and full 80-tick horizons. The only configuration
change was `env_kwargs.sim_config.scene_config.cpu_workers: 0 -> 2`.

| Metric | CPU0 A1 / A2 | CPU2 B1 / B2 |
|---|---:|---:|
| Command wall s | 15.363 / 14.351 | 13.603 / 13.589 |
| Evaluation wall s | 11.835 / 10.829 | 10.429 / 10.428 |
| Evaluation episodes/h | 6083.838 / 6648.836 | 6903.633 / 6904.341 |
| Evaluation steps/s | 135.196 / 147.752 | 153.414 / 153.430 |
| Rollout s | 4.046997 / 3.847167 | 4.638201 / 4.735975 |
| Step/observation s | 3.526140 / 3.485817 | 4.104140 / 4.167298 |
| RPC mean ms | 2.948 / 1.945 | 3.027 / 3.221 |
| GPU busy mean % | 14.903 / 14.897 | 16.036 / 15.091 |
| VRAM mean / peak GiB | 2.381/4.914; 2.382/4.914 | 2.668/4.914; 2.675/4.914 |
| Accounted CPU cores | 0.832 / 0.841 | 0.969 / 0.963 |
| Success once / unsuccessful | 12/8; 12/8 | 12/8; 12/8 |
| Worker failure / OOM | 0/0; 0/0 | 0/0; 0/0 |

The faster command/evaluation clocks come from lower construction time and are
excluded from the optimization decision. CPU2 sustained rollout throughput is
**0.842118x**, or 18.75% more rollout time. Its second-group-only ratio is 0.844010x.
Step throughput is 0.847731x; second-group-only is 0.842638x. The setting is rejected
and was not expanded to 100 episodes. All four strict 20-record comparisons pass;
no per-step trajectory equality is claimed for these untraced runs.

## Review of other candidates

| Candidate | Evidence and decision |
|---|---|
| Smaller PhysX pair buffers | Earlier reported formal gain was 1.23x, dominated by lifecycle work; not retained as a sustained step optimization. |
| Environment reuse with seeded reset | Earlier reported formal gain was 1.28x, dominated by repeated construction/cleanup; excluded from this sustained-runtime goal. |
| IK tensor indexing | Earlier approximately 3.8% screen was unstable in warmed groups; rejected. |
| GC freezing | No eligible sustained improvement established; not retained. |
| Jacobian-only CUDA graph | Not implemented. Eligible profiles cover only 12.07% of the R12 matrix's enclosing step span; the actual Jacobian fraction is smaller. Bridge uses iterative target IK and does not enter that branch. |
| Feature packing / isolated synchronization | Insufficient measured budget. Removing all RollBall feature preparation caps rollout at 1.003x; all RPC removal caps it at 1.101x and includes inference. |
| Bridge observation/physics audit | Its four R12 episodes require 30,000 physics microsteps, 300 native image/segmentation observations and 720 contact-pair queries. State fetch already occurs once per control, not once per microstep; background decoding is constructor-only. Dropping steps, segmentation, success queries or moving-camera updates changes the evaluation inputs/rules. |
| CPU dispatcher 0 -> 2 | New controlled ABBA screen above: rejected at 0.842118x rollout throughput. |

Historical gains above are not rerun measurements, are not multiplied, and do not
enter a current speedup claim. R12's 66-profile first-episode matrix is a traced
correctness workload, not a warmed formal benchmark. Its 556.037 s rollout proxy
contains 415.068 s step/observations (74.65%), 132.791 s RPC (23.88%) and 2.866 s
feature preparation (0.52%). Bridge is 28.46% of that rollout. Even halving the
entire Bridge step region would give only about 1.11x on this specific mix; a
safe independent change with that gain has not been demonstrated.

No additional runtime change with a stable, meaningful sustained gain was found
in this bounded investigation beyond the already completed parallel/autotune
work. This does not establish that future simulator changes are impossible.
No performance patch or new performance PR is justified by these results, and
issue #31's full-profile 2x target is not met. Integration correctness changes and
new model-side references in #99 are excluded from performance gains.

## Evidence

[Raw baseline, trial and profiler reports with checksums](https://github.com/QLL-CCDD/RoboScope-Kit/releases/tag/maniskill-issue25-native-review-20261111)
are external release assets: `issue31-rollball-diagnostics-20261111.tar.gz`,
16,857,452 bytes, SHA-256
`971bb4f74219d45f12f5d338d546da04ea368ebc4d4e2ba960af6239a15aeb5f`.
Independent member/content and SQLite audits pass. The archive retains failed
counter access and the built-in capture's empty CUDA coverage; missing counters
are not represented as zero.


All raw profiler reports/SQLite remain physically under
`/root/autodl-tmp/maniskill-issue25-review-20261109/issue31-20261111`.
The adjacent `rollball-current-metrics.json` records clocks, distributions, source identities,
resource scopes and receipt hashes; `current-rollball-analysis.json` retains source records and
SQL-derived per-process context details. Owned collectors and CSV summaries are outside Git.
No GPU workload was launched by the CPU-only extraction/statistics scripts.
