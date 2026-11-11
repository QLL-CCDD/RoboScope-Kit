# Opt-in Jacobian CUDA graphs

On 2026-11-11, a bounded default-off simulator patch reduces repeated small-kernel
dispatch in the unchanged `pytorch-kinematics==0.7.4` Jacobian. The preceding
[method diagnosis](RUNTIME_DIAGNOSTICS.md#granular-sustained-step-diagnosis)
places the complete Jacobian at 21-24% of sustained RollBall/Bridge wall time.
This is an independent runtime improvement for validated configurations;
**issue #31's full-profile 2x target remains unmet**. The incomplete #25
acceptance and deferred model/assets are tracked in #99 separately.

## Implementation and scope

`build-runtime.sh` checksums the original/patched ManiSkill Kinematics and the
helper, applies `patches/jacobian-cuda-graphs.patch` in a clean source build, and
records the patch in build provenance. No installed environment is edited by hand.
The patch keeps the original Jacobian, FP32 arithmetic and multiplication order;
immutable topology indices and constants become capture-compatible. It changes
neither Policy code, weights, inference, physics, cameras, inputs, control rates,
episode seeds/horizons, scoring nor termination checks.

Only the checksum-pinned Panda `panda_hand_tcp` (N10, seven arm joints) and
WidowX `ee_gripper_link` (N1, six joints) are enabled. Other geometry/batch widths,
CPU, gradient-bearing inputs, autocast, noncontiguous inputs and nondefault
locations retain the original path. Enable the patch explicitly:

```bash
export ROBOSCOPE_MANISKILL_JACOBIAN_GRAPHS=1
# Run the normal benchmark command with its original manifest and Policy config.
```

The switch defaults to `0`. The cache contains at most two entries per Worker;
keys include pinned URDF/end link, device, shape, dtype, precision, inference mode
and CUDA stream. Each entry owns only pure static chain geometry and graph buffers.
Every call copies the complete current joint tensor and clones returned tensors.
It does not retain a scene, articulation, environment or episode state. The current
Worker calls serially on an immutable chain; same-stream concurrent Python-thread
transactions and post-capture geometry/device mutation are outside this scope.
Capture failure is explicit because a failed capture can invalidate CUDA state.

## Fast screening and independent fixed comparison

The machine reports an RTX 4090 with 49,140 MiB VRAM and driver 595.71.05;
the runtime sees a 20-core CPU budget. Measured Kit code is b58,
`b0a47c92480ddcbd041fb2af3cbbf41cc571152d`; subsequent report edits are Markdown only.
The official FP32 RollBall checkpoint SHA-256 is
`fb2b81ce700125bc189d540571fda66f858c84af06becff43dbe0142951a86ac`.
K1/G1/N10, the original Panda `pd_ee_delta_pos`, state inputs, 100/20 Hz,
80 control ticks, `success_once`, and seeds are fixed. No MPS, autotune,
profiler or trajectory hashing runs inside these timed commands.

The first graph implementation captures again after every environment rebuild.
Its four 20-episode ABBA runs give only **1.0072x rollout** despite a much faster
isolated Jacobian. It is rejected without a 100-episode expansion. V2 retains
only the pure graph/geometry across independently recreated environments:
four 20-episode ABBA runs give 1.1693x rollout and 1.0968x evaluation, with
12/20 successes and exact records each. This passes the expansion gate.

V2 then runs six fresh commands, **A/B/B/A/A/B**, 100 episodes each, seeds 0-99.
A disables the graph; B enables it. All groups perform their original complete
seeded reset and simulation. The following are medians of three commands per
condition; brackets give the observed minimum and maximum. Compilation/capture
is included in the first rollout, evaluation and total clocks.

| Metric | Original path A | Graph B |
|---|---:|---:|
| Total command s | 39.543 [39.040, 40.069] | 35.713 [35.016, 35.753] |
| Evaluation s | 35.715 [35.276, 36.105] | 31.864 [31.235, 31.883] |
| Evaluation episodes/h | 10,079.86 [9,970.97, 10,205.37] | 11,297.88 [11,291.26, 11,525.51] |
| Evaluation lane steps/s | 224.00 [221.58, 226.79] | 251.06 [250.92, 256.12] |
| Summed serial rollout s | 19.997 [19.772, 20.018] | 15.790 [15.727, 15.903] |
| Step including observation s | 17.924 [17.644, 17.969] | 13.796 [13.631, 13.916] |
| Policy startup/load s | 1.569 | 1.542 |
| Runtime preparation s | 0.602 | 0.603 |
| Environment creation s, ten calls | 7.288 | 7.394 |
| Complete seeded reset s, ten calls | 4.055 | 4.017 |
| Policy RPC s, 800 batches | 1.824 | 1.745 |
| Policy RPC mean ms/batch | 2.280 | 2.182 |
| Environment close s, ten calls | 2.251 | 2.277 |
| Cold first-group rollout median s | 2.236 | 2.031 |
| Warm group rollout p50 / p95 s | 1.972 / 2.017 | 1.530 / 1.602 |
| GPU busy command / evaluation / common-window % | 27.22 / 29.95 / 33.83 | 28.59 / 32.27 / 37.77 |
| Command mean / peak device VRAM GiB | 3.891 / 4.914 | 3.786 / 4.961 |
| Common-window mean device VRAM GiB | 4.652 | 4.662 |
| Accounted process-tree CPU cores | 1.031 | 1.029 |
| Success / unsuccessful episodes, each command | 63 / 37 | 63 / 37 |
| Actual lane steps, each command | 8,000 | 8,000 |
| Missing / duplicate / Worker failures / OOM | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Median evaluation throughput is **1.1208x**, total throughput **1.1073x**,
rollout **1.2664x**, and step **1.2992x**. Ratios of mean times are respectively
1.1275x, 1.1143x, 1.2608x and 1.2950x. Creation and reset are not the source
of the retained gain. Warmed group improvement persists as groups accumulate.
Per-phase medians need not sum to a median command. Ten lanes share a group's
wall time: group quantiles above pool 27 warmed groups per condition and are
not 270 independent episode-latency observations.

Evaluation is the runtime's `evaluation_s` phase: it includes Worker startup,
environment lifecycle, rollout and queue drain. Total is measured subprocess
entry-to-exit and includes Policy loading, imports, preparation and result/cleanup.
There is no separately instrumented standalone interpreter or file-writing clock.
Rollout/step/creation/reset are serial Worker CPU wall intervals, not isolated
GPU durations. RPC includes serialization, server inference and scheduling wait;
server GPU latency and queue-only wait are unavailable. State evaluation consumes
no rendered frame; isolated render API/Vulkan durations are unmeasured here.
SM/Tensor/RT counters remain **N/A** after actual `ERR_NVGPUCTRPERM`; NVML busy
is not a substitute. NVML is device-wide, sampled every 250 ms; common-window
boundaries also have 200 ms completion-poll uncertainty. These are observations
under the tested bounds, not a physical GPU-utilization ceiling.

Every B command has exactly one capture, taking 0.300-0.481 s and reporting
8,473,088 allocated / 25,165,824 reserved additional bytes at capture.
The device-wide sampled peak rises by 48 MiB. Peaks stay constant across the
remaining nine groups in all three B runs. Process RSS rises about 9 MiB over
those groups in both conditions; this bounded workload does not prove unlimited
lifetime memory stability. Full-horizon truncation is normal in all 100 records,
not a runtime timeout.

## Correctness and multi-task checks

All six timed commands have identical 100-record identities, initial-state and
observation digests, first actions, returns, success results and actual terminal
masks, excluding only elapsed time. This is not a per-step assertion for all 100.
A separate traced 20-episode full80 RollBall comparison checks **1,600 actions
and 1,620 observations** at zero tolerance across environment group recreation.
An actual Octo/WidowX `PutEggplantInBasketScene-v1` comparison checks **10 actions
and 11 observations** in an explicitly diagnostic prefix of its original
120-tick profile; cameras/segmentation and all 1,000 physical microsteps remain.
These diagnostic timings are excluded from speed ratios.

The pure-kinematics lifetime probe additionally rebuilds each robot's chain three
times with different full inputs. All 18 J+pose and six J-only calls are
`torch.equal`; old output tensors remain unchanged and at most two entries live.
The immutable source/install audit verifies all 340 regular frozen source files,
94 installed SDK files, the installed helper/patched Kinematics, manifest and b58.
The fresh build/self-test, 270 existing Linux pytest cases, Ruff, Python support,
`bash -n` and ShellCheck pass. No repository tests are added.

Current independent native GPU-vectorized references also run 100 episodes at
N10 with the graph disabled/enabled. Each matches the corresponding SDK's 100
records at zero action/return tolerance: 63 successes, 8,000 steps, no discrepancies.
Native command times are 33.895/30.861 s, reported evaluation intervals
30.967/27.830 s and import-inclusive formal intervals 32.431/29.290 s.
These are one reference each with different clock boundaries from the SDK,
not a repeated SDK-versus-native speedup claim. An ordinary real-model
`--quick-test` with the graph requested at unsupported Panda N1 completes its
original 80 ticks and records zero graph captures, verifying the original path.

## Retuning after the runtime change

Four additional commands use the existing diagonal strategy, N10, the same
100 scored seeds/80 ticks, and identical K<=2/G<=2 bounds and budgets. Each
candidate supplies three complete vector groups per Worker: 30 probe episodes
for K1 and 60 for K2. Each command tests K1/G1, K2/G1 and K2/G2; actual two-Worker
health is verified. Probes and the final 100 scored episodes are separate.

| Mode | Selected K/G | Total s | Tuning s | Evaluation s | Episodes/h | Lane steps/s |
|---|---|---:|---:|---:|---:|---:|
| Autotune, graph off | 2/1 | 84.740 | 55.977 | 26.363 | 13,655.68 | 303.46 |
| Autotune, graph on | 2/2 | 76.868 | 51.227 | 23.175 | 15,533.84 | 345.20 |
| Autotune + Policy MPS, graph off | 2/1 | 83.647 | 55.924 | 25.012 | 14,393.37 | 319.85 |
| Autotune + Policy MPS, graph on | 2/1 | 78.967 | 52.856 | 23.395 | 15,388.17 | 341.96 |

| Mode | GPU busy command / evaluation / common % | Mean / peak VRAM GiB | CPU cores | RPC mean ms/batch |
|---|---:|---:|---:|---:|
| Autotune, graph off | 42.12 / 54.98 / 65.54 | 5.613 / 9.217 | 1.745 | 2.404 |
| Autotune, graph on | 39.31 / 53.54 / 66.07 | 5.332 / 9.311 | 1.733 | 2.633 |
| Autotune + Policy MPS, graph off | 41.15 / 57.00 / 66.58 | 5.560 / 9.242 | 1.726 | 2.884 |
| Autotune + Policy MPS, graph on | 38.98 / 51.97 / 67.60 | 5.222 / 9.336 | 1.719 | 3.150 |

Each row is one completed retune, not three repeats. All four scored cohorts have
63/100 successes, 8,000 steps and identical records, including masks and returns;
only their named-suite label differs from the fixed-run suite. There are zero
missing/duplicate records, Worker failures or OOMs. Tuning includes the initial
model loads and probes; the subsequent load phase can reuse a ready server.
Its cost remains in total time. Counter/render/queue limitations are as above.

The original-path tuner rejects G2 with +0.3% gain, while graph-on selects it
at +4.6%; the MPS runs reject G2 at -4.8%/+1.1%. Different selected G means
the ordinary retuned ratio cannot isolate the graph or MPS switches. The fixed
three-repeat comparison above establishes the independent runtime benefit.
Policy MPS applies to the inference replicas, not the simulator's CUDA context
or Vulkan renderer. The highest observed common-window mean is 67.60% within
these bounds. It is not a hardware ceiling or proof that MPS is faster.

## Remaining hotspot review

The retained change reduces repeated Jacobian dispatch at every eligible control
step. The other measured candidates were reviewed again before stopping:

| Candidate | Measured scope and decision |
|---|---|
| PhysX native stepping | 42-45% of the earlier sustained windows, including submission, GPU execution and blocking `fetchResults(true)`. Required substeps and fresh post-step observations prevent deleting or deferring this region without changing physics or action/input ordering. No supported equivalent persistent change is established. |
| More CPU PhysX dispatch threads | Existing controlled 0-to-2-thread ABBA slows rollout to 0.8421x; rejected. |
| Additional state fetch removal | Already one fetch per control, only 0.333%/0.033% of the measured RollBall/Bridge windows; negligible ceiling. |
| Isolated FK or target-pose shortcut | Roughly 4% overall ceiling even if the region becomes twice as fast; prior IK-index screen is inconsistent. No second speculative patch. |
| Observation/render skipping | Policies consume each required observation; Bridge also needs segmentation/contact queries and moving-camera updates. Removing them changes inputs or scoring. |
| RPC/feature packing | Features cover about 0.5%; Policy GPU execution is tiny. RPC wall includes required inference and transport, so deleting it is not an eligible optimization. |
| Smaller pair pools / environment reuse / startup caching | Earlier lifecycle gains are diluted by longer rollouts and do not satisfy the sustained-step goal. Not combined with this patch. |
| Parallel/autotune/MPS | Existing scheduler is reused and retuned separately above; no new scheduler change or multiplied cross-configuration speedup. |

This bounded review finds no further evidence-backed change with a clear,
material sustained benefit ready to implement. It does not establish that future
native-engine work is impossible. No further unmotivated large runs are needed.

## Reproduction and evidence

Build an empty runtime using the regular builder and verified asset/source caches.
Fix `envs_per_worker=10` in a copy of the manifest and select the unchanged
`rollball_state` profile through a one-profile suite. Use the same suite and
Policy configuration for both conditions:

```bash
ROBOSCOPE_MANISKILL_JACOBIAN_GRAPHS=0 /path/to/runtime/env/bin/python \
  -m roboscope_kit.runtime.benchmark --benchmark maniskill \
  --benchmark-manifest /path/to/matched-manifest.json --suite matched_rollball \
  --checkpoint-source /path/to/RollBall-v1.pt --policy-config /path/to/policy.json \
  --episodes 100 --concurrency 1 --replicas 1 --output-dir /path/to/new-off-result
# Repeat with switch=1 and a new output directory, preserving every other setting.
```

Raw trial, trajectory, model-reference, tuning, build and audit artifacts remain
outside Git under `issue31-graph-cache-b58-20261111`. The frozen source archive
SHA-256 is `00c00fcf0c4e6ae7ccf10e4d86e224d349a9ad58a9b2df43a830d57b56aed0d5`.
Autotune/MPS and native-reference results are reported separately from the fixed
comparison; no historical gain is multiplied into these numbers.
