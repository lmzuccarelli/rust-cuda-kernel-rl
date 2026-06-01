use crate::config::load::Parameters;
use crate::inference::prompts::{
    get_available_optimizations, get_best_optimization_prompt, get_combined, get_optimization_plan,
    get_performance_state_category, get_profile_prompt, get_state_match_prompt,
    get_task_generate_code_prompt,
};
use crate::kernel::profile::{Profile, ProfileInterface};
use crate::utils::common::{extract_code, extract_code_all};
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
    async fn execute_flow(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct Controller {}

#[derive(Serialize, Deserialize, Clone, Debug)]
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

    async fn execute_flow(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        for item in parameters.workflow_batch.iter() {
            log::info!("[execute_flow] item {}", item);
            // ensure logs directory is created for each cuda-kernel
            fs::create_dir_all(format!("logs/{}/rl-ncu/baseline", item))?;

            let mut elapsed_cycles = 0_u64;
            let mut code = String::new();
            let mut ncu_report = String::new();
            let mut state = String::new();
            let mut json_plan = String::new();

            log::trace!("[execute_flow] initial elapsed_cycles {}", elapsed_cycles);
            log::trace!("[execute_flow] initial code           {}", code);
            log::trace!("[execute_flow] initial ncu_report     {}", ncu_report);
            log::trace!("[execute_flow] initial state          {}", state);
            log::trace!("[execute_flow] initial json_plan      {}", json_plan);

            match parameters.test {
                false => {
                    let x = parameters.flow_control;
                    let baseline_dir = format!("logs/{}/rl-ncu/baseline", item);
                    let payload = format!(
                        r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}" }}"##,
                        item,
                        parameters.working_dir.clone(),
                        parameters.gpu_arch,
                        "rl-ncu/baseline"
                    );

                    if x & 2u8 == 2 {
                        // get all baseline artifacts first
                        // first call the compile endpoint
                        log::info!("[execute_flow] baseline calling compile cuda kernel endpoint");
                        let url = format!("{}/v1/compile", parameters.compile_server_url);
                        let file_name = format!("{}/compile.txt", baseline_dir);
                        process_post_call(Some(file_name), url, payload.clone()).await?;

                        // download cuda kernel (init.cu)
                        log::info!("[execute_flow] baseline downloading cuda kernel");
                        let url = format!("{}/v1/cuda-kernel", parameters.compile_server_url);
                        let file_name = format!("{}/init.cu", baseline_dir);
                        code = process_post_call(Some(file_name), url, payload.clone()).await?;
                    } else {
                        code = fs::read_to_string(format!("{}/init.cu", baseline_dir))?;
                    }

                    if x & 4u8 == 4 {
                        // call the execute endpoint
                        log::info!("[execute_flow] baseline calling execute cuda kernel endpoint");
                        let url = format!("{}/v1/execute", parameters.gpu_server_url);
                        let file_name = format!("{}/execute.txt", baseline_dir);
                        process_post_call(Some(file_name), url, payload.clone()).await?;
                    }

                    if x & 8u8 == 8 {
                        // call the nvidia ncu profile endpoint
                        log::info!("[execute_flow] baseline calling profile cuda kernel endpoint");
                        let url = format!("{}/v1/profile", parameters.gpu_server_url);
                        let file_name = format!("{}/profile.txt", baseline_dir);
                        ncu_report = process_post_call(Some(file_name), url, payload).await?;
                    } else {
                        ncu_report = fs::read_to_string(format!("{}/profile.txt", baseline_dir))?;
                    }

                    elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;
                    log::info!("[execute_flow] baseline elapsed_cycles {}", elapsed_cycles);

                    if x & 16u8 == 16 {
                        log::info!(
                            "[execute_flow] baseline calling llm endpoint (current state profile)"
                        );
                        // set the initial prompt for the llm
                        let prompt =
                            get_profile_prompt(code.clone(), ncu_report.clone()).replace("\n", "");
                        // call the llm endpoint
                        let url = format!("{}/v1/prompt", parameters.llm_server_url);
                        let file_name = format!("{}/llm_state_response.txt", baseline_dir);
                        state = process_post_call(Some(file_name), url, prompt).await?;
                    } else {
                        state =
                            fs::read_to_string(format!("{}/llm_state_response.txt", baseline_dir))?;
                    }

                    if x & 32u8 == 32 {
                        log::info!(
                            "[execute_flow] baseline calling llm endpoint (optimization plan)"
                        );

                        // get the 9 top matching optimizations in json format from the llm
                        let avail_opt = get_available_optimizations();
                        let prompt_op = get_optimization_plan(
                            parameters.max_trajectories - 1,
                            state.clone(),
                            code.clone(),
                            avail_opt,
                        );
                        // call the llm endpoint
                        let url = format!("{}/v1/prompt", parameters.llm_server_url);
                        let file_name = format!("{}/optimization-plan.json", baseline_dir);
                        json_plan = process_post_call(Some(file_name), url, prompt_op).await?;
                    } else {
                        json_plan =
                            fs::read_to_string(format!("{}/optimization-plan.json", baseline_dir))?;
                    }

                    if x & 64u8 == 64 {
                        let start = Instant::now();
                        log::info!("[execute_flow] executing rollout");
                        let json_plan_updated = json_plan.replace("```json", "");
                        let plans =
                            serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;
                        let category = Profile::get_category(state)?;
                        let mut count = 0;
                        let mut futs = FuturesUnordered::new();
                        for plan in plans.iter() {
                            count += 1;
                            // Generate a shorter 8-character ID (6 bytes)
                            let short = short_id_with_bytes(6)?;
                            let trajectory_dir =
                                format!("logs/{}/rl-ncu/trajectory_{}_{}", item, count, short);
                            log::info!(
                                "[execute_flow] executing trajectory {} for technique {}",
                                trajectory_dir,
                                plan.technique.to_owned()
                            );
                            fs::create_dir_all(trajectory_dir.to_owned())?;
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
                                format!("{}/{}.prompt", trajectory_dir, plan.technique),
                                task_prompt.clone(),
                            )?;

                            let url = format!("{}/v1/prompt", parameters.llm_server_url);
                            // this will always start with step 0, then within the complex function
                            // each new step (until max.rollout) will be added
                            let file_name = format!(
                                "{}/step_0_{}_llm_response.txt",
                                trajectory_dir, plan.technique
                            );
                            futs.push(process_post_call(Some(file_name), url, task_prompt));
                        }
                        // Wait for the remaining to finish.
                        while let Some(response) = futs.next().await {
                            log::debug!("{}", &response.unwrap()[..100]);
                        }

                        let elapsed = start.elapsed();
                        log::info!("[execute flow] completed rollout in {:?}", elapsed);
                    }

                    /*
                    // we are trying to get the ONE technique that will yield the largest performance gain
                    log::info!("[execute_flow] baseline calling llm endpoint (state match)");
                    // set the initial prompt for the llm
                    let prompt = get_state_match_prompt(state.clone()).replace("\n", "");
                    // call the llm endpoint
                    let url = format!("{}/v1/prompt", parameters.llm_server_url);
                    process_post_call(
                        item.to_string(),
                        url,
                        "baseline/llm_match_response.txt".to_string(),
                        prompt,
                    )
                    .await?;

                    log::info!("[execute_flow] calling llm endpoint (baseline best optimization)");
                    // set the initial prompt for the llm
                    let prompt = get_best_optimization_prompt(state, avail_opt).replace("\n", "");
                    // call the llm endpoint
                    let url = format!("{}/v1/prompt", parameters.llm_server_url);
                    process_post_call(
                        item.to_string(),
                        url,
                        "baseline/llm_optimization_response.txt".to_string(),
                        prompt,
                    )
                    .await?;

                    log::info!(
                        "[execute_flow] executing rollout initializing {} trajectories",
                        parameters.max_trajectories
                    );
                    for i in 1..parameters.max_trajectories {
                        // Generate a shorter 8-character ID (6 bytes)
                        let short = short_id_with_bytes(6)?;
                        let trajectory_dir = format!("logs/{}/trajectory_{}_{}", item, i, short);
                        log::info!(
                            "[execute_flow] rollout creating trajectory directory {} ",
                            trajectory_dir
                        );
                        fs::create_dir_all(trajectory_dir)?;
                    }
                    */
                }
                true => {
                    log::warn!("[execute_flow] testing current flow control");
                    let ncu_report =
                        fs::read_to_string(format!("logs/{}/rl-ncu/baseline/profile.txt", item))?;
                    elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;
                    log::info!("[execute_flow] testing elapsed_cycles {}", elapsed_cycles);
                    let (result, reward) = Profile::calculate_improvement(elapsed_cycles, 7677773)?;
                    log::info!(
                        "[execute_flow] testing result {} : reward {}",
                        result,
                        reward
                    );
                    let json_plan = fs::read_to_string(format!(
                        "logs/{}/rl-ncu/baseline/optimization-plan.json",
                        item
                    ))?;
                    let state = fs::read_to_string(format!(
                        "logs/{}/rl-ncu/baseline/llm_state_response.txt",
                        item
                    ))?;

                    let category = Profile::get_category(state)?;
                    log::info!("[execute_flow] testing category {}", category);

                    println!();

                    let json_plan_updated = json_plan.replace("```json", "");
                    let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;
                    for op in plans.iter() {
                        log::info!("[execute_flow] testing technique   : {}", op.technique);
                        log::info!(
                            "[execute_flow] testing relevance   : {}",
                            op.relevance_score
                        );
                        log::info!("[execute_flow] testing description : {}", op.description);
                        println!();
                    }

                    println!();

                    let input = fs::read(format!(
                        "logs/{}/rl-ncu/trajectory_1__KMbpkXv/step_0_shared_memory_tiling_llm_response.txt",
                        item
                    ))?;
                    let result = extract_code(String::from_utf8(input)?)?;
                    log::info!("{}", result);

                    println!();

                    extract_code_all(
                        item.to_string(),
                        parameters.working_dir.clone(),
                        format!("{}/v1/upload", parameters.compile_server_url),
                        parameters.gpu_arch,
                    )
                    .await?;
                }
            }
        }
        log::info!("[execute_flow] workflow controller completed successfully");
        Ok(())
    }
}
