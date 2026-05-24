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
