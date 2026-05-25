pub fn get_profile_prompt(code: String, ncu_report: String) -> String {
    format!(
        r#"
You are a GPU performance analysis expert. Analyze this NVIDIA NSight Compute (NCU) profiling report and provide a qualitative summary of the kernel's performance state.

CODE IMPLEMENTATION:
```cpp
{code}
```

NCU REPORT:
{ncu_report}  

Provide your analysis in this EXACT format:

PERFORMANCE_SIGNATURE: [2-3 sentence summary of what is limiting performance and the overall execution pattern]

RELATIVE_PATTERNS:
- memory_pressure: [very_low|low|moderate|high|very_high]
- compute_utilization: [very_low|low|moderate|high|very_high] 
- access_patterns: [excellent|good|moderate|poor|very_poor]
- cache_efficiency: [excellent|good|moderate|poor|very_poor]
- occupancy_level: [very_low|low|moderate|high|very_high]
- parallelism_utilization: [very_low|low|moderate|high|very_high]
- specialied_hw_usage: [very_low|low|moderate|high|very_high]
- [List 3-5 key secondary performance characteristics]
- [Focus on patterns you observe in the data]
- [Include cache behavior, memory access patterns, occupancy]
- [Note any resource conflicts or inefficiencies]

PRIMARY_BOTTLENECK: [memory_bound|compute_bound|latency_bound|hybrid_bound]

//code signature: loop pattern /branches(in summary/generic)

CONTEXT_DESCRIPTION: [Brief description of the workload characteristics and optimization opportunities]

Focus on qualitative patterns and relationships rather than specific numbers. Look for the underlying performance characteristics that drive behavior.
"#
    )
}

pub fn get_state_match_prompt(state: String) -> String {
    format!(
        r#"You are a GPU optimization expert. Compare the current kernel performance state against known optimization states and find the best match.

CURRENT STATE TO MATCH:
{state}

Context: 

KNOWN OPTIMIZATION STATES:

STATE: high_level_inefficiency
Primary Bottleneck: Instructions uncessary for computing the kernel
Secondary Characteristics: Excessive or redundant compute operations, memory accesses. Poor scaling, inefficient program logic. Can have misleadingly high utilization.
Available Optimizations:
  - algorithmic_changes: 1.00x speedup
  - dynamic_programming: 1.00x speedup
  - data_structure_optimizations: 1.00x speedup


STATE: memory_bandwidth_saturated
Primary Bottleneck: Global memory bandwidth saturation
Secondary Characteristics: Memory throughput >80%, Bandwidth utilization high, Coalescing efficiency varies
Available Optimizations:
  - vectorized_memory_access: 1.00x speedup
  - memory_coalescing_optimization: 1.00x speedup
  - shared_memory_tiling: 1.00x speedup


STATE: memory_latency_bound
Primary Bottleneck: Memory access latency and cache misses
Secondary Characteristics: Memory throughput 40-70%, High cache miss rate, Random access patterns
Available Optimizations:
  - shared_memory_blocking: 1.00x speedup
  - register_blocking: 1.00x speedup
  - prefetching_strategies: 1.00x speedup


STATE: memory_bank_conflicts
Primary Bottleneck: Shared memory bank conflicts causing serialization
Secondary Characteristics: Shared memory bank conflicts detected, Irregular access patterns in shared memory
Available Optimizations:
  - bank_conflict_padding: 1.00x speedup
  - access_pattern_swizzling: 1.00x speedup
  - broadcast_optimization: 1.00x speedup


STATE: cache_inefficient
Primary Bottleneck: Cache misses and inefficient cache utilization
Secondary Characteristics: Low L1/L2 cache hit rates, Spatial locality issues, Poor temporal reuse
Available Optimizations:
  - cache_blocking_algorithms: 1.00x speedup
  - temporal_locality_optimization: 1.00x speedup
  - l2_cache_persistence: 1.00x speedup


STATE: compute_throughput_saturated
Primary Bottleneck: Instruction throughput and arithmetic unit saturation
Secondary Characteristics: Compute throughput >80%, All execution units busy, High arithmetic intensity
Available Optimizations:
  - tensor_core_utilization: 1.00x speedup
  - instruction_level_parallelism: 1.00x speedup
  - fast_math_optimization: 1.00x speedup


STATE: instruction_mix_suboptimal
Primary Bottleneck: Poor instruction scheduling and functional unit utilization
Secondary Characteristics: Unbalanced use of functional units, Some units idle while others saturated
Available Optimizations:
  - instruction_scheduling_optimization: 1.00x speedup
  - functional_unit_balancing: 1.00x speedup
  - loop_fusion_optimization: 1.00x speedup


STATE: thread_divergence_high
Primary Bottleneck: SIMT efficiency loss due to divergent execution paths
Secondary Characteristics: High warp divergence, Branching in inner loops, Irregular control flow
Available Optimizations:
  - predicated_execution: 1.00x speedup
  - branch_reduction_techniques: 1.00x speedup
  - warp_level_primitives: 1.00x speedup


STATE: low_occupancy_register_pressure
Primary Bottleneck: Register pressure limiting number of active warps
Secondary Characteristics: Low occupancy due to high register usage, Register spills detected
Available Optimizations:
  - register_pressure_reduction: 1.00x speedup
  - variable_scoping_optimization: 1.00x speedup
  - spill_elimination: 1.00x speedup


STATE: low_occupancy_shared_memory
Primary Bottleneck: Shared memory usage limiting number of resident blocks
Secondary Characteristics: Low occupancy due to shared memory limits, High shared memory usage per block
Available Optimizations:
  - shared_memory_optimization: 1.00x speedup
  - dynamic_shared_memory: 1.00x speedup
  - shared_memory_reuse: 1.00x speedup


STATE: insufficient_parallelism
Primary Bottleneck: Not enough parallel work to hide latency
Secondary Characteristics: Low warp utilization, Small problem sizes, Insufficient work per SM
Available Optimizations:
  - work_per_thread_increase: 1.00x speedup
  - thread_coarsening: 1.00x speedup
  - persistent_kernel_pattern: 1.00x speedup


STATE: hybrid_bound
Primary Bottleneck: Balanced memory-compute limitations (hybrid)
Secondary Characteristics: Memory throughput 40-70% and compute throughput 40-70%, indicating that neither memory nor compute dominates. Latency hiding is moderate and improvement is possible from both sides.
Available Optimizations:
  - memory_compute_overlap: 1.00x speedup
  - adaptive_block_sizing: 1.00x speedup
  - algorithmic_optimization: 1.00x speedup


STATE: memory_compute_balanced
Primary Bottleneck: Balanced memory and compute load with room for optimization in both
Secondary Characteristics: Memory throughput 40-60%, Compute throughput 40-60%, Both contribute to bottleneck
Available Optimizations:
  - memory_compute_overlap: 1.00x speedup
  - adaptive_tiling: 1.00x speedup
  - fused_operations: 1.00x speedup


STATE: latency_memory_bound
Primary Bottleneck: Both occupancy and memory access patterns need optimization
Secondary Characteristics: Low occupancy combined with memory issues, Multiple bottlenecks present
Available Optimizations:
  - occupancy_memory_codesign: 1.00x speedup
  - hierarchical_tiling: 1.00x speedup
  - resource_balanced_design: 1.00x speedup


STATE: api_overhead_dominant
Primary Bottleneck: CUDA API overhead and kernel launch latency
Secondary Characteristics: Small kernels with high launch overhead, Frequent CPU-GPU synchronization
Available Optimizations:
  - kernel_fusion: 1.00x speedup
  - cuda_graphs: 1.00x speedup
  - persistent_kernels: 1.00x speedup


STATE: transfer_bandwidth_limited
Primary Bottleneck: Host-device transfer bandwidth and latency
Secondary Characteristics: High PCIe transfer overhead, Large data movement, Host-device bottleneck
Available Optimizations:
  - transfer_optimization: 1.00x speedup
  - data_compression: 1.00x speedup
  - zero_copy_memory: 1.00x speedup

MATCHING INSTRUCTIONS:
1. Primary bottleneck must align (memory_bound with memory_bound, etc.)
2. Look for similar secondary characteristics and patterns
3. Consider the performance signature and context similarity
4. Focus on qualitative patterns rather than exact matches

Provide your analysis in this EXACT format:

BEST_MATCH: [state_name from STATE: (as indicated in the context above) or "NEW_STATE_NEEDED"]
CONFIDENCE: [0.0 to 1.0]
REASONING: [Explain why this state matches, focusing on bottleneck alignment and similar characteristics]

If confidence < 0.6, respond with BEST_MATCH: NEW_STATE_NEEDED
"#
    )
}

pub fn get_best_optimization_prompt(state_summary: String) -> String {
    format!(
        r#"
You are a GPU optimisation expert. A kernel has been analysed and its qualitative performance characteristics are shown below. From the list of available optimisation techniques pick the ONE technique that you judge will yield the largest performance gain. 
Respond STRICTLY in the format:

BEST_OPTIMIZATION: <technique name>
REASONING: <brief rationale>

CURRENT STATE SUMMARY
{state_summary}

AVAILABLE OPTIMISATIONS:
STATE: high_level_inefficiency
  - algorithmic_changes (pred 1.00x | conf 0.5): Replace the current algorithm with a more efficient approach
  - dynamic_programming (pred 1.00x | conf 0.5): Utilize memory to avoid redundant computation
  - data_structure_optimizations (pred 1.00x | conf 0.5): Change underlying data structures/representations

STATE: memory_bandwidth_saturated
  - vectorized_memory_access (pred 1.00x | conf 0.5): Use float4/int4/half2 for wider transactions
  - memory_coalescing_optimization (pred 1.00x | conf 0.5): Align access patterns to cache lines
  - shared_memory_tiling (pred 1.00x | conf 0.5): Cache frequently accessed data on-chip
  - data_layout_transformation (pred 1.00x | conf 0.5): Convert AoS to SoA layouts
  - compression_techniques (pred 1.00x | conf 0.5): Use mixed precision (FP16) to halve bandwidth

STATE: memory_latency_bound
  - shared_memory_blocking (pred 1.00x | conf 0.5): Tile data to fit in shared memory
  - register_blocking (pred 1.00x | conf 0.5): Keep frequently used data in registers
  - prefetching_strategies (pred 1.00x | conf 0.5): Software prefetching for predictable patterns
  - cache_aware_algorithms (pred 1.00x | conf 0.5): Restructure for better locality
  - texture_memory_usage (pred 1.00x | conf 0.5): Use texture cache for read-only data

STATE: memory_bank_conflicts
  - bank_conflict_padding (pred 1.00x | conf 0.5): Add padding to shared memory arrays
  - access_pattern_swizzling (pred 1.00x | conf 0.5): Use XOR patterns to distribute accesses
  - broadcast_optimization (pred 1.00x | conf 0.5): Structure for broadcast reads when possible
  - memory_layout_permutation (pred 1.00x | conf 0.5): Reorder data to avoid conflicts

STATE: cache_inefficient
  - cache_blocking_algorithms (pred 1.00x | conf 0.5): Structure working sets for cache sizes
  - temporal_locality_optimization (pred 1.00x | conf 0.5): Reorder computations for reuse
  - l2_cache_persistence (pred 1.00x | conf 0.5): Mark critical data for L2 retention
  - read_only_cache_optimization (pred 1.00x | conf 0.5): Use __ldg() for read-only data

STATE: compute_throughput_saturated
  - tensor_core_utilization (pred 1.00x | conf 0.5): Use Tensor Cores for matrix operations
  - instruction_level_parallelism (pred 1.00x | conf 0.5): Unroll loops and reorder operations
  - fast_math_optimization (pred 1.00x | conf 0.5): Use fast math flags and intrinsics
  - specialized_instruction_usage (pred 1.00x | conf 0.5): Use GPU-specific intrinsics
  - vectorized_operations (pred 1.00x | conf 0.5): Use vector data types and operations
  - SIMD_operations (pred 1.00x | conf 0.5): Use packed SIMD datatypes such as half2

STATE: instruction_mix_suboptimal
  - instruction_scheduling_optimization (pred 1.00x | conf 0.5): Reorder for better pipeline usage
  - functional_unit_balancing (pred 1.00x | conf 0.5): Balance load across different units
  - loop_fusion_optimization (pred 1.00x | conf 0.5): Combine kernels to balance instruction mix
  - mathematical_transformation (pred 1.00x | conf 0.5): Adjust operators to solve the problem in a different way

STATE: thread_divergence_high
  - predicated_execution (pred 1.00x | conf 0.5): Replace branches with conditional arithmetic
  - branch_reduction_techniques (pred 1.00x | conf 0.5): Use lookup tables and bit manipulation
  - warp_level_primitives (pred 1.00x | conf 0.5): Use shuffle and vote functions
  - stream_compaction (pred 1.00x | conf 0.5): Filter and compact data to reduce divergence
  - work_distribution_optimization (pred 1.00x | conf 0.5): Sort data to group similar work

STATE: low_occupancy_register_pressure
  - register_pressure_reduction (pred 1.00x | conf 0.5): Reduce variable scope and reuse
  - variable_scoping_optimization (pred 1.00x | conf 0.5): Limit variable lifetimes
  - spill_elimination (pred 1.00x | conf 0.5): Restructure to avoid local memory usage
  - launch_bounds_tuning (pred 1.00x | conf 0.5): Use __launch_bounds__ to control registers
  - algorithmic_register_reduction (pred 1.00x | conf 0.5): Change algorithm to use fewer registers

STATE: low_occupancy_shared_memory
  - shared_memory_optimization (pred 1.00x | conf 0.5): Reduce per-block shared memory usage
  - dynamic_shared_memory (pred 1.00x | conf 0.5): Use dynamic allocation for flexibility
  - shared_memory_reuse (pred 1.00x | conf 0.5): Reuse buffers for multiple phases
  - block_size_adaptation (pred 1.00x | conf 0.5): Adjust block size to optimize occupancy

STATE: insufficient_parallelism
  - work_per_thread_increase (pred 1.00x | conf 0.5): Assign more work to each thread
  - thread_coarsening (pred 1.00x | conf 0.5): Process multiple elements per thread
  - persistent_kernel_pattern (pred 1.00x | conf 0.5): Use long-running kernels with work queues
  - grid_size_optimization (pred 1.00x | conf 0.5): Ensure sufficient blocks for latency hiding
  - cooperative_groups (pred 1.00x | conf 0.5): Use grid-wide synchronization

STATE: hybrid_bound
  - memory_compute_overlap (pred 1.00x | conf 0.5): Pipeline memory and compute operations to hide latency
  - adaptive_block_sizing (pred 1.00x | conf 0.5): Adjust thread-block sizes dynamically to balance resource usage
  - algorithmic_optimization (pred 1.00x | conf 0.5): Restructure algorithms for better memory-compute balance
  - fused_operations (pred 1.00x | conf 0.5): Fuse memory-bound and compute-bound kernels to reduce intermediate traffic
  - SIMD_operations (pred 1.00x | conf 0.5): Use packed SIMD datatypes (e.g., half2) to improve both bandwidth and compute throughput

STATE: memory_compute_balanced
  - memory_compute_overlap (pred 1.00x | conf 0.5): Pipeline memory and compute operations
  - adaptive_tiling (pred 1.00x | conf 0.5): Dynamically adjust tile sizes for balance
  - fused_operations (pred 1.00x | conf 0.5): Combine memory-bound and compute-bound kernels
  - algorithmic_rebalancing (pred 1.00x | conf 0.5): Choose algorithms that balance both aspects

STATE: latency_memory_bound
  - occupancy_memory_codesign (pred 1.00x | conf 0.5): Jointly optimize occupancy and memory
  - hierarchical_tiling (pred 1.00x | conf 0.5): Multi-level tiling for both occupancy and locality
  - resource_balanced_design (pred 1.00x | conf 0.5): Balance all resource constraints
  - workload_restructuring (pred 1.00x | conf 0.5): Fundamental algorithm changes

STATE: api_overhead_dominant
  - kernel_fusion (pred 1.00x | conf 0.5): Combine multiple small kernels
  - cuda_graphs (pred 1.00x | conf 0.5): Use graphs to batch operations
  - persistent_kernels (pred 1.00x | conf 0.5): Long-running kernels with internal queues
  - asynchronous_execution (pred 1.00x | conf 0.5): Overlap with streams
  - dynamic_parallelism (pred 1.00x | conf 0.5): Launch child kernels from GPU

STATE: transfer_bandwidth_limited
  - transfer_optimization (pred 1.00x | conf 0.5): Use pinned memory and async transfers
  - data_compression (pred 1.00x | conf 0.5): Compress data before transfer
  - zero_copy_memory (pred 1.00x | conf 0.5): Direct GPU access to host memory
  - unified_memory_optimization (pred 1.00x | conf 0.5): Use managed memory with hints
  - pipeline_parallelism (pred 1.00x | conf 0.5): Overlap transfers with computation

"#
    )
}
