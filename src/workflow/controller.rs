use crate::config::load::Parameters;
use crate::inference::prompts::{
    get_available_optimizations, get_combined, get_optimization_plan,
    get_performance_state_category, get_profile_prompt, get_state_match_prompt,
    get_task_generate_code_prompt,
};
use crate::kernel::profile::{Profile, ProfileInterface};
use crate::utils::common::{extract_code, extract_code_all, find_cuda_file, pick_weighted};
use crate::workflow::api_client::{process_get_call, process_post_call};
use custom_logger as log;
use futures::stream::FuturesUnordered;
use futures::stream::StreamExt;
use serde_derive::{Deserialize, Serialize};
use short_id::short_id_with_bytes;
use std::fs;
use std::time::Instant;

pub trait ControllerInterface {
    async fn get_health(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
    async fn execute_baseline_flow(
        paramaters: Parameters,
    ) -> Result<(), Box<dyn std::error::Error>>;
    async fn execute_agent_flow(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct Controller {}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, PartialOrd)]
pub struct OptimizationPlan {
    pub technique: String,
    pub relevance_score: f32,
    pub description: String,
}

impl ControllerInterface for Controller {
    async fn get_health(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        // check compiler endpoint
        let res_compiler =
            process_get_call(format!("{}/v1/health", parameters.compile_server_url)).await?;
        log::info!("{}", res_compiler);
        // check gpu endpoint
        let res_gpu = process_get_call(format!("{}/v1/health", parameters.gpu_server_url)).await?;
        log::info!("{}", res_gpu);
        // check llm endpoint
        let res_llm = process_get_call(format!("{}/v1/health", parameters.llm_server_url)).await?;
        log::info!("{}", res_llm);
        Ok(())
    }

    async fn execute_baseline_flow(
        parameters: Parameters,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for item in parameters.workflow_batch.iter() {
            log::info!("[execute_baseline_flow] item {}", item);
            // ensure logs directory is created for each cuda-kernel
            fs::create_dir_all(format!("logs/{}/rl-ncu/baseline", item))?;

            let mut elapsed_cycles = 0_u64;
            let mut code = String::new();
            let mut ncu_report = String::new();
            let mut state = String::new();
            let mut match_state = String::new();
            let mut json_plan = String::new();

            log::trace!(
                "[execute_baseline_flow] initial elapsed_cycles {}",
                elapsed_cycles
            );
            log::trace!("[execute_baseline_flow] initial code           {}", code);
            log::trace!(
                "[execute_baseline_flow] initial ncu_report     {}",
                ncu_report
            );
            log::trace!("[execute_baseline_flow] initial state          {}", state);
            log::trace!(
                "[execute_baseline_flow] initial match_state          {}",
                match_state
            );
            log::trace!(
                "[execute_baseline_flow] initial json_plan      {}",
                json_plan
            );

            let x = parameters.flow_control;
            let baseline_dir = format!(
                "{}/out/{}/rl-ncu/baseline",
                parameters.working_dir.to_owned(),
                item
            );
            let payload = format!(
                r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}" }}"##,
                item, parameters.working_dir, parameters.gpu_arch, baseline_dir
            );
            // as we are saving locally to replay buffer , change out to logs
            let local_baseline_dir = baseline_dir.replace("/out/", "/logs/");

            if x & 2u8 == 2 {
                // call the compile endpoint
                log::info!("[execute_baseline_flow] baseline calling compile cuda kernel endpoint");
                let url = format!("{}/v1/compile", parameters.compile_server_url);
                let file_name = format!("{}/compile.txt", local_baseline_dir);
                process_post_call(Some(file_name), url, payload.clone()).await?;

                // download cuda kernel (init.cu)
                log::info!("[execute_baseline_flow] baseline downloading cuda kernel");
                let url = format!("{}/v1/cuda-kernel", parameters.compile_server_url);
                let file_name = format!("{}/init.cu", local_baseline_dir);
                code = process_post_call(Some(file_name), url, payload.clone()).await?;
            } else {
                code = fs::read_to_string(format!("{}/init.cu", local_baseline_dir))?;
            }

            if x & 4u8 == 4 {
                // call the execute endpoint
                log::info!("[execute_baseline_flow] baseline calling execute cuda kernel endpoint");
                let url = format!("{}/v1/execute", parameters.gpu_server_url);
                let file_name = format!("{}/execute.txt", local_baseline_dir);
                process_post_call(Some(file_name), url, payload.clone()).await?;
            }

            if x & 8u8 == 8 {
                // call the nvidia ncu profile endpoint
                log::info!("[execute_baseline_flow] baseline calling profile cuda kernel endpoint");
                let url = format!("{}/v1/profile", parameters.gpu_server_url);
                let file_name = format!("{}/profile.txt", local_baseline_dir);
                ncu_report = process_post_call(Some(file_name), url, payload).await?;
            } else {
                ncu_report = fs::read_to_string(format!("{}/profile.txt", local_baseline_dir))?;
            }

            elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;
            log::info!(
                "[execute_baseline_flow] baseline elapsed_cycles {}",
                elapsed_cycles
            );

            if x & 16u8 == 16 {
                log::info!(
                    "[execute_baseline_flow] baseline calling llm endpoint (current state profile)"
                );
                // set the initial prompt for the llm
                let prompt = get_profile_prompt(code.clone(), ncu_report.clone()).replace("\n", "");
                // call the llm endpoint
                let url = format!("{}/v1/prompt", parameters.llm_server_url);
                let file_name = format!("{}/llm_state_response.txt", local_baseline_dir);
                state = process_post_call(Some(file_name), url, prompt).await?;
            } else {
                state =
                    fs::read_to_string(format!("{}/llm_state_response.txt", local_baseline_dir))?;
            }

            if x & 32u8 == 32 {
                log::info!("[execute_baseline_flow] baseline calling llm endpoint (match state)");
                // set the initial prompt for the llm
                let prompt = get_state_match_prompt(state.clone());
                // call the llm endpoint
                let url = format!("{}/v1/prompt", parameters.llm_server_url);
                let file_name = format!("{}/llm_match_state_response.txt", local_baseline_dir);
                match_state = process_post_call(Some(file_name), url, prompt).await?;
            } else {
                match_state = fs::read_to_string(format!(
                    "{}/llm_match_state_response.txt",
                    local_baseline_dir
                ))?;
            }

            if x & 64u8 == 64 {
                log::info!(
                    "[execute_baseline_flow] baseline calling llm endpoint (optimization plan)"
                );

                // get the top_n matching optimizations in json format from the llm
                let avail_opt = get_available_optimizations();
                let prompt_op = get_optimization_plan(
                    parameters.max_rollout,
                    state.clone(),
                    code.clone(),
                    avail_opt,
                );
                // call the llm endpoint
                let url = format!("{}/v1/prompt", parameters.llm_server_url);
                let file_name = format!("{}/optimization-plan.json", local_baseline_dir);
                json_plan = process_post_call(Some(file_name), url, prompt_op).await?;
            } else {
                json_plan =
                    fs::read_to_string(format!("{}/optimization-plan.json", local_baseline_dir))?;
            }

            if x & 128u8 == 128 {
                let start = Instant::now();
                log::info!("[execute_baseline_flow] executing rollout");
                let json_plan_updated = json_plan.replace("```json", "");
                let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;
                let category = Profile::get_category(state)?;
                //let mut count = 0;
                let mut futs = FuturesUnordered::new();
                for x in 0..parameters.max_rollout {
                    let plan = pick_weighted(plans.clone())?;
                    // Generate a shorter 8-character ID (6 bytes)
                    let short = short_id_with_bytes(6)?;
                    let trajectory_dir = format!(
                        "{}/logs/{}/rl-ncu/trajectory_{}_{}",
                        parameters.working_dir,
                        item,
                        x + 1,
                        short
                    );
                    log::info!(
                        "[execute_baseline_flow] executing trajectory {} for technique {}",
                        trajectory_dir,
                        plan.technique.to_owned()
                    );
                    fs::create_dir_all(format!("{}/step_0", trajectory_dir.to_owned()))?;
                    let state_category = get_performance_state_category();
                    let combined = get_combined(state_category)?;
                    let task_prompt = get_task_generate_code_prompt(
                        plan.technique.to_owned(),
                        category.to_owned(),
                        plan.description.to_owned(),
                        ncu_report.to_owned(),
                        code.to_owned(),
                        combined,
                    );

                    fs::write(
                        format!("{}/step_0/{}.prompt", trajectory_dir, plan.technique),
                        task_prompt.clone(),
                    )?;

                    let url = format!("{}/v1/prompt", parameters.llm_server_url);
                    // this will always start with step 0, then within the complex flow
                    // each new step (until max.rollout) will be added
                    let file_name = format!(
                        "{}/step_0/{}_llm_response.txt",
                        trajectory_dir, plan.technique
                    );
                    futs.push(process_post_call(Some(file_name), url, task_prompt));
                }
                // Wait for the remaining to finish.
                while let Some(response) = futs.next().await {
                    match response {
                        Ok(_) => log::info!("[execute_baseline_flow] call succeeded"),
                        Err(e) => log::error!("[execute_baseline_flow] call failed {}", e),
                    }
                }

                let dir = format!("{}/out/{}/rl-ncu", parameters.working_dir, item);
                let url = format!("{}/v1/upload", parameters.compile_server_url);
                extract_code_all(
                    dir,
                    parameters.working_dir.to_owned(),
                    url,
                    parameters.gpu_arch,
                )
                .await?;

                let elapsed = start.elapsed();
                log::info!("[execute_baseline_flow] completed rollout in {:?}", elapsed);
            }
        }
        log::info!("[execute_baseline_flow] workflow controller completed successfully");
        Ok(())
    }

    async fn execute_agent_flow(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: loop through all trajectories
        for item in parameters.workflow_batch.iter() {
            // get baseline state and elapsed_cycles
            // optimization plans were calculated in the baseline run
            // no optimization plan (json) file is found in the step_0 directories
            let baseline_dir = format!("{}/logs/{}/rl-ncu/baseline", parameters.working_dir, item);
            let baseline_ncu_report = fs::read_to_string(format!("{}/profile.txt", baseline_dir))?;
            let mut ncu_report = baseline_ncu_report.clone();
            let baseline_elapsed_cycles = Profile::get_elapsed_cycles(baseline_ncu_report.clone())?;
            let current_trajectory = "trajectory_1_mINMOfqW";

            for step in parameters.rollout_start..parameters.max_rollout {
                // plan_count starts at 9 and then decrements by one until we reach 4
                let mut plan_count = parameters.max_rollout - (step + 1);
                if plan_count <= 4 {
                    plan_count = 4;
                }
                let local_target_dir = format!(
                    "{}/logs/{}/rl-ncu/{}/step_{}",
                    parameters.working_dir, item, current_trajectory, step
                );
                let target_dir = local_target_dir.replace("/logs/", "/out/");

                // read code
                // TODO: fix for multiple cuda files in same directory
                let cuda_file = find_cuda_file(local_target_dir.clone())?;
                let code = fs::read_to_string(format!("{}/{}", local_target_dir, cuda_file))?;
                let payload = format!(
                    r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_name": "{}" , "code": {:?} }}"##,
                    item, parameters.working_dir, parameters.gpu_arch, target_dir, cuda_file, code
                );

                // upload kernel
                let url = format!("{}/v1/upload", parameters.compile_server_url);
                process_post_call(None, url.clone(), payload.clone()).await?;
                log::info!("[execute_agent_flow] target_dir {}", target_dir);
                log::info!("[execute_agent_flow] using kernel {}", cuda_file);

                // compile kernel
                log::info!("[execute_agent_flow] calling compile cuda kernel endpoint",);
                let url = format!("{}/v1/compile", parameters.compile_server_url);
                let file_name = format!("{}/compile.txt", local_target_dir);

                // verify compile, kernel execution and profiling
                // if there are any errors we refer back to the baseline
                // so that we can continue to the next step
                // TODO: refactor for legibility
                let res = process_post_call(Some(file_name), url, payload.clone()).await;
                match res {
                    Ok(_) => {
                        // execute the kernel to ensure it passes our basic test
                        log::info!("[execute_agent_flow] calling execute cuda kernel endpoint");
                        let url = format!("{}/v1/execute", parameters.gpu_server_url);
                        let file_name = format!("{}/execute.txt", local_target_dir);
                        let exec_res =
                            process_post_call(Some(file_name), url, payload.clone()).await;
                        match exec_res {
                            Ok(_) => {
                                // call the nvidia ncu profile endpoint
                                log::info!(
                                    "[execute_agent_flow] calling profile cuda kernel endpoint"
                                );
                                let url = format!("{}/v1/profile", parameters.gpu_server_url);
                                let file_name = format!("{}/profile.txt", local_target_dir);
                                // update ncu_report (override previous)
                                let profile_res =
                                    process_post_call(Some(file_name), url, payload.clone()).await;
                                match profile_res {
                                    // update to the new profile else use the baseline
                                    Ok(profile) => ncu_report = profile,
                                    Err(e) => log::error!("[execute_agent_flow] profile {}", e),
                                }

                                let elapsed_cycles =
                                    Profile::get_elapsed_cycles(ncu_report.clone())?;

                                // calculate improvement
                                log::info!(
                                    "[execute_agent_flow] elapsed cycles : {}",
                                    elapsed_cycles
                                );
                                let (perc, reward) = Profile::calculate_improvement(
                                    baseline_elapsed_cycles,
                                    elapsed_cycles,
                                )?;
                                log::info!(
                                    "[execute_agent_flow] percentage improvement {}% : reward {}",
                                    perc,
                                    reward
                                );
                                let mut contents = format!(
                                    "baseline elapsed cycles : {}\n",
                                    baseline_elapsed_cycles
                                );
                                contents.push_str(&format!(
                                    "elapsed cycles          : {}\n",
                                    elapsed_cycles
                                ));
                                contents
                                    .push_str(&format!("improvement             : {}%\n", perc));
                                contents
                                    .push_str(&format!("reward                  : {}\n", reward));
                                fs::write(format!("{}/stats.txt", local_target_dir), contents)?;
                                if perc < -100.0 {
                                    log::error!(
                                        "[execute_agent_flow] reward degradation is too severe"
                                    );
                                    ncu_report = baseline_ncu_report.clone();
                                }
                            }
                            Err(e) => {
                                log::error!("[execute_agent_flow] execute kernel {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("[execute_agent_flow] compile {}", e);
                    }
                }

                // get new state prompt based on the ncu_report
                // if previous steps fail it will use the baseline ncu_report
                log::info!("[execute_agent_flow] calling llm (state)");
                let state_prompt =
                    get_profile_prompt(code.clone(), ncu_report.clone()).replace("\n", "");
                // call the llm endpoint
                let url = format!("{}/v1/prompt", parameters.llm_server_url);
                let file_name = format!("{}/llm_state_response.txt", local_target_dir);
                let state = process_post_call(Some(file_name), url, state_prompt).await?;

                // get the top_n matching optimizations in json format from the llm
                let avail_opt = get_available_optimizations();
                let prompt_op =
                    get_optimization_plan(plan_count, state.clone(), code.clone(), avail_opt);

                // call llm for optimization plan
                log::info!("[execute_agent_flow] calling llm (optimization plan)");
                let url = format!("{}/v1/prompt", parameters.llm_server_url);
                let file_name = format!("{}/optimization-plan.json", local_target_dir);
                let json_plan = process_post_call(Some(file_name), url, prompt_op).await?;
                let json_plan_updated = json_plan.replace("```json", "").replace("```", "");
                let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;
                let category = Profile::get_category(state.clone())?;
                // pick the weighted plan
                let plan = pick_weighted(plans.clone())?;
                let state_category = get_performance_state_category();
                let combined = get_combined(state_category)?;
                let task_prompt = get_task_generate_code_prompt(
                    plan.technique.clone(),
                    category.to_owned(),
                    plan.description,
                    ncu_report.to_owned(),
                    code.to_owned(),
                    combined,
                );

                if step == parameters.max_rollout - 1 {
                    // no use working on the max_rollout
                    log::info!(
                        "[execute_agent_flow] max_rollout reached : exiting flow gracefully"
                    );
                    break;
                }

                // setup for the next step
                let local_target_dir = format!(
                    "{}/logs/{}/rl-ncu/{}/step_{}",
                    parameters.working_dir,
                    item,
                    current_trajectory,
                    step + 1
                );
                fs::create_dir_all(local_target_dir.clone())?;

                fs::write(
                    format!("{}/{}.prompt", local_target_dir, plan.technique),
                    task_prompt.clone(),
                )?;

                log::info!("[execute_agent_flow] calling llm (task generate code)");
                let url = format!("{}/v1/prompt", parameters.llm_server_url);
                let file_name = format!("{}/{}_llm_response.txt", local_target_dir, plan.technique);
                let contents = process_post_call(Some(file_name), url, task_prompt).await?;

                // extract code
                let code = extract_code(contents)?;
                fs::write(format!("{}/{}.cu", local_target_dir, plan.technique), code)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // this brings everything from parent's scope into this scope
    use super::*;
    use crate::config::load::{ConfigInterface, ImplConfigInterface};
    use regex::Regex;
    use std::fs;

    #[tokio::test]
    async fn test_all() -> Result<(), Box<dyn std::error::Error>> {
        let impl_config = ImplConfigInterface {};
        let res_params = impl_config.read("config/application-config.json".to_owned());
        let parameters = match res_params {
            Ok(params) => params,
            Err(e) => {
                eprintln!("[main] config file error: {}", e);
                std::process::exit(1);
            }
        };

        let item = "level1/001_Square_matrix_multiplication";
        log::warn!("[execute_baseline_flow] testing current flow control");
        let ncu_report = fs::read_to_string(format!("logs/{}/rl-ncu/baseline/profile.txt", item))?;
        let elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;
        log::info!(
            "[execute_baseline_flow] testing elapsed_cycles {}",
            elapsed_cycles
        );
        let (result, reward) = Profile::calculate_improvement(elapsed_cycles, 7677773)?;
        log::info!(
            "[execute_baseline_flow] testing result {} : reward {}",
            result,
            reward
        );
        let json_plan = fs::read_to_string(format!(
            "logs/{}/rl-ncu/trajectory_1_mINMOfqW/step_0/optimization-plan.json",
            item
        ))?;
        let state = fs::read_to_string(format!(
            "logs/{}/rl-ncu/baseline/llm_state_response.txt",
            item
        ))?;

        let category = Profile::get_category(state)?;
        log::info!("[execute_baseline_flow] testing category {}", category);

        println!();

        let json_plan_updated = json_plan.replace("```json", "").replace("```", "");
        let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;
        for op in plans.iter() {
            log::info!(
                "[execute_baseline_flow] testing technique   : {}",
                op.technique
            );
            log::info!(
                "[execute_baseline_flow] testing relevance   : {}",
                op.relevance_score
            );
            log::info!(
                "[execute_baseline_flow] testing description : {}",
                op.description
            );
            println!();
        }

        println!();

        for _x in 0..10 {
            let pick = pick_weighted(plans.clone())?;
            log::info!(
                "[execute_baseline_flow] testing weighted plan {} {}",
                pick.technique,
                pick.relevance_score
            );
        }

        let base_dir = format!(
            "{}/logs/{}/rl-ncu/trajectory_1_mINMOfqW/step_0",
            parameters.working_dir, item
        );
        let cuda_file = find_cuda_file(base_dir)?;
        log::info!("[execute_baseline_flow] testing cuda file {}", cuda_file);

        let _kernel_test = r#"
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void matmul_wmma_kernel(
"#;
        let re = Regex::new("[_]{2}global[_]{2}[_a-zA-Z()\\s]*\\svoid\\s([a-zA-Z0-9_]*)")?;
        let mut kernel = fs::read_to_string(
            "/home/lzuccarelli/Projects/rust-cuda-kernel-rl/logs/level1/001_Square_matrix_multiplication/rl-ncu/trajectory_1_mINMOfqW/step_3/tensor_core_utilization.cu",
        )?;

        let mut kernel_name;
        println!("testing regex");
        for cap in re.captures_iter(&kernel) {
            kernel_name = cap[1].to_string();
            println!("[run] profiling : kernel {}", kernel_name);
        }

        kernel = fs::read_to_string(
            "/home/lzuccarelli/Projects/rust-cuda-kernel-rl/logs/level1/001_Square_matrix_multiplication/rl-ncu/baseline/init.cu",
        )?;
        println!("testing regex");
        for cap in re.captures_iter(&kernel) {
            kernel_name = cap[1].to_string();
            println!("[run] profiling : kernel {}", kernel_name);
        }

        Ok(())
    }
}
