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

## Repository locations

The repository is organized by purpose.

1.  `btb/` contains the baseline BTB implementations and custom
    replacement policies.
2.  `inc/` contains simulator headers, including `inc/ooo_cpu.h`.
3.  `src/` contains simulator source code, including the CPU
    implementation.
4.  `dpc3_traces/` contains the compressed ChampSim traces used in the
    experiments.
5.  `launch/` contains experiment launch scripts. `launch/launch.sh` and
    `launch/run_exp.sh` are entry points, and
    `launch/scripts/createConfig.sh` creates experiment configurations.
6.  `btb_experiment_results/` contains experiment logs under
    `btb_experiment_results/logs/`, the run manifest
    `btb_experiment_results/run_manifest.csv`, and the plotting script
    `btb_experiment_results/visualize.py`.
7. `analysis_25M/` contains aggregated CSV metrics and plots for the 25
    million instruction runs.
8. `analysis_50M/` contains aggregated CSV metrics and plots for the 50
    million instruction runs.
9. `bin/` contains compiled ChampSim executables for the configured
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

Trace files are stored in `dpc3_traces/`. Confirm that the requested
trace exists and that the executable name matches a binary in `bin/`.

The launch scripts in `launch/` support running groups of experiments.
Review the scripts before execution to confirm the selected policies,
trace list, and instruction counts.

## Collecting and visualizing results

## Collecting and Visualizing Results

The `launch/run_exp.sh` script is used to execute experiments across different BTB models and traces. It runs multiple experiments concurrently using background tasks, with the current configuration set to **4 simultaneous jobs**.

The results are stored in directories such as `results_25M/` and `results_50M/`, depending on the simulation length.

To collect, process, and visualize the experimental results, run `launch/visualize.py`. The script processes the results from the corresponding results directory and generates analysis outputs. (Change the directory in the script to folder of result eg: results_25M)

The generated analysis is organized by simulation length:

1. `analysis_25M/` contains summary CSV files and plots for 25 million instruction runs.
2. `analysis_50M/` contains summary CSV files and plots for 50 million instruction runs.

Each analysis directory contains the following summary files:

1. `comparison_metrics_by_model.csv`
2. `comparison_metrics_per_trace.csv`
3. `ipc_geomean_speedup.csv`
4. `metric_coverage.csv`
5. `metrics_per_run.csv`

The `plots/` subdirectory contains visualizations for overall model comparisons, per trace performance, and priority and ordering experiments.

## Evaluation metrics

The project records several metrics to understand both performance and
branch behavior.

1.  **IPC:** instructions per cycle. Higher IPC indicates greater
    instruction throughput.
2.  **Geometric mean IPC speedup:** the geometric mean of per trace IPC
    ratios relative to the `convBTB` baseline.
3.  **Branch accuracy:** the accuracy value reported by the simulator.
4.  **Branch mispredictions:** the reported number of branch
    mispredictions.
5.  **Branch MPKI:** branch misses per thousand
    instructions.
6.  **Cache miss rates:** the reported L1D, L2C, and LLC miss rates,
    included to help check whether broader memory behavior changes
    alongside BTB policy changes.

## References

This project uses ChampSim and builds on the BTBX research artifact. 
