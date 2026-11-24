# BRB: Online Behavior Aware BTB Replacement

BRB is a ChampSim based project that investigates online branch behavior
signals for Branch Target Buffer (BTB) replacement. The objective is to
determine whether branch frequency, branch type, target behavior, and
reuse information can help make better BTB eviction decisions than a
conventional policy.

## Motivation

The BTB stores branch target information to support instruction
fetching. Since the BTB has finite capacity, inserting a new entry may
require evicting an existing entry.

This project evaluates alternative replacement policies that use information observed during simulation,
without relying on separate offline profiling.

The main signals investigated are:

1.  **Access frequency:** how often a branch is encountered.
2.  **Branch type priority:** whether calls, returns, conditional
    branches, or indirect branches should receive different retention
    priorities.
3.  **Target behavior:** whether a branch continues to resolve to a
    consistent target.
4.  **Reuse distance:** how much branch activity occurs between uses of
    an entry.
5.  **Replacement ordering:** how the order in which these signals are
    considered affects victim selection.

## Implemented BTB variants

The policy implementations are located in `btb/`. The main baseline is
`btb/convBTB.btb`. Other variants include `btb/BTBX.btb`,
`btb/pdede.btb`, and custom policies for branch type priority, LFU,
target behavior, reuse distance, and combinations of these signals.

Examples of custom policy files include:

1.  `btb/call_first.btb`
2.  `btb/conditional_first.btb`
3.  `btb/indirect_first.btb`
4.  `btb/return_first.btb`
5.  `btb/lfu.btb`
6.  `btb/lfu_priority.btb`
7.  `btb/lfu_priority_target.btb`
8.  `btb/lfu_priority_target_reuse_d.btb`
9.  `btb/reuse_d_priority.btb`
10. `btb/target_lfu_priority.btb`
11. `btb/srrip_priority.btb`
12. `btb/ghost_utility.btb`

## Checkpoint 2 update

Checkpoint 2 expands the replacement-policy comparison with two SRRIP-based policies:

### `srrip_priority`

`btb/srrip_priority.btb` combines Static RRIP (SRRIP) victim selection
with branch-type priority and observed frequency:

- Hits set the entry's Re-reference Prediction Value (RRPV) to `0`,
  indicating near-term reuse.
- Indirect branches, indirect calls, and returns enter protected with
  RRPV `0`; conditional and direct branches enter with RRPV `3`.
- When entries are considered for eviction, the policy first looks for
  RRPV `3` entries. If multiple candidates are present, it selects the
  one with the lowest observed frequency.
- If no RRPV `3` entry exists, it ages entries by incrementing their
  RRPV.
- Branch hit frequency is capped at 15.

This policy replaces the conventional exact-LRU-style victim choice with
a bounded re-reference value while retaining branch-aware insertion and
frequency information.

### `ghost_utility`

`btb/ghost_utility.btb` combines SRRIP with online reuse and target-stability
utility tracking:

- Reuse utility increases when a BTB entry receives hits.
- Stability utility increases when an observed branch target matches the
  previously recorded target and decreases after a target change.
- Utility is calculated as
  `(2 * reuse_score + stability_score) / 3`.
- Branch priority is added to indirect branches, indirect calls, and
  returns.
- A ghost hit and/or higher utility assigns a lower insertion RRPV,
  protecting branches that are estimated to be valuable.
- A 512-entry ghost table remembers evicted branches for up to 100,000
  BTB counter events. Re-insertion can recover the evicted branch's
  utility and receive additional reuse priority.

SRRIP aging is used when no entry has reached maximum RRPV. On eviction,
the ghost table records the evicted branch IP and its current utility.

### Checkpoint 2 experiment configuration

The new checkpoint 2 experiment set compares the conventional `convBTB`
baseline, `lfu_priority_target`, `srrip_priority`, and `ghost_utility`.
It uses the following Google-workload traces:

1. `arizona_0001.champsim-003.gz`
2. `arizona_0002.champsim-001.gz`
3. `charlie_0000.champsim-005.gz`
4. `charlie_0002.champsim-003.gz`
5. `merced_0000.champsim-002.gz`
6. `merced_0004.champsim-001.gz`
7. `sierra.a.3_0000.champsim-001.gz`
8. `sierra.a.6_0001.champsim-001.gz`
9. `tahoe_0001.champsim-002.gz`
10. `tahoe_0003.champsim-001.gz`
11. `tango_0000.champsim-004.gz`
12. `tango_0001.champsim-003.gz`

`launch/run_exp.sh` builds and runs these models in parallel batches.
The completed run is visible in the
`results_50M/` directory.

## Repository locations

The repository is organized by purpose.

1.  `btb/` contains the baseline BTB implementations and custom
    replacement policies.
2.  `inc/` contains simulator headers, including `inc/ooo_cpu.h`.
3.  `src/` contains simulator source code, including the CPU
    implementation.
4.  `dpc3_traces/` contains the compressed ChampSim traces used in the
    experiments.
5. `analysis_25M/` contains aggregated CSV metrics and plots for the 25
    million instruction runs.
6. `analysis_50M/` contains aggregated CSV metrics and plots for the 50
    million instruction runs.
7. `bin/` contains compiled ChampSim executables for the configured
    policies.

## Experimental configuration

The main configuration used for comparison is:

  Component                   Setting
  --------------------------- ---------------------------
  Branch predictor            `hashed_perceptron`
  L1 instruction prefetcher   `fdip`
  L2 cache prefetcher         `next_line`
  LLC prefetcher              `spp_dev`
  LLC replacement argument    `no`
  L1D replacement             `lru`
  Number of cores             `1`
  Warmup length               `50` million instructions
  Measurement length          `50` million instructions

Experiments compare the conventional `convBTB` baseline against the BTB
variants using the same workload and simulator settings. The trace set
used in the current analysis includes `client_001`, `client_005`,
`server_001`, `server_010`, `spec_gcc_001`, `spec_gobmk_001`,
`spec_perlbench_001`, and `spec_x264_001`.

## Building ChampSim

Run build commands from the repository root.

Build the conventional BTB configuration:

``` bash
./build_champsim.sh hashed_perceptron convBTB fdip next_line spp_dev no lru 1
```

Build BTBX:

``` bash
./build_champsim.sh hashed_perceptron BTBX fdip next_line spp_dev no lru 1
```

Build PDede:

``` bash
./build_champsim.sh hashed_perceptron pdede fdip next_line spp_dev no lru 1
```

Build SRRIP+Priority policy:

``` bash
./build_champsim.sh hashed_perceptron srrip_priority fdip next_line spp_dev no lru 1
```

Build ghost-utility policy:

``` bash
./build_champsim.sh hashed_perceptron ghost_utility fdip next_line spp_dev no lru 1
```

To build without FDIP, replace the third argument `fdip` with `no`. For
example:

``` bash
./build_champsim.sh hashed_perceptron convBTB no next_line spp_dev no lru 1
```

The available binary names can be checked in `bin/`. The repository
includes binaries for multiple BTB policies and configurations.

## Running a simulation

The following example runs the conventional BTB configuration for the
`server_010` trace with 50 million instructions of warmup and 50 million
instructions of measurement:

``` bash
./run_champsim.sh hashed_perceptron-convBTB-fdip-next_line-spp_dev-no-lru-1core 50 50 server_010.champsimtrace.xz
```
The launch scripts in `launch/` support running groups of experiments.
Review the scripts before execution to confirm the selected policies,
trace list, and instruction counts.

## Collecting results

The `launch/run_exp.sh` script is used to execute experiments across different BTB models and traces. It runs multiple experiments concurrently using background tasks, with the current configuration set to **6 simultaneous jobs**.

The results are stored in directories such as `results_25M/` and `results_50M/`, depending on the simulation length.


## Evaluation metrics

The project records several metrics to understand both performance and
branch behavior.

1.  **IPC:** instructions per cycle. Higher IPC indicates greater
    instruction throughput.
2.  **Geometric mean IPC speedup:** the geometric mean of per trace IPC
    ratios relative to the `convBTB` baseline.
3.  **Branch accuracy:** the accuracy value reported by the simulator.
4.  **Branch MPKI:** branch misses per thousand
    instructions.

## References

This project uses ChampSim and builds on the BTBX research artifact. 
