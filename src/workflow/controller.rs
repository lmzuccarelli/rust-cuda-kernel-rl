use crate::config::load::Parameters;
use crate::inference::prompts::{
    get_available_optimizations, get_combined, get_optimization_plan,
    get_performance_state_category, get_profile_prompt, get_state_match_prompt,
    get_task_generate_code_prompt,
};
use crate::kernel::profile::{Profile, ProfileInterface};
use crate::utils::common::{
    extract_code, extract_code_from_call, find_cuda_file, find_most_performant_kernel,
    get_trajectories, pick_weighted,
};
use crate::workflow::api_client::{process_get_call, process_post_call};
use custom_logger as log;
// Due to rate limiting executing the agent in parallel is not used for now
// use futures::stream::FuturesUnordered;
// use futures::stream::StreamExt;
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
            fs::create_dir_all(format!(
                "logs/{}/{}/rl-ncu/baseline",
                parameters.llm_model, item
            ))?;

            let mut elapsed_cycles = 0_i64;
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
                "{}/out/{}/{}/rl-ncu/baseline",
                parameters.working_dir.to_owned(),
                parameters.llm_model,
                item
            );
            let payload = format!(
                r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}" }}"##,
                item, parameters.working_dir, parameters.gpu_arch, baseline_dir
            );
            // as we are saving locally to replay buffer , change out to logs
            let local_baseline_dir = baseline_dir.replace("/out/", "/logs/");

            if x & 1u8 == 1 {
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

            if x & 2u8 == 2 {
                // call the execute endpoint
                log::info!("[execute_baseline_flow] baseline calling execute cuda kernel endpoint");
                let url = format!("{}/v1/execute", parameters.gpu_server_url);
                let file_name = format!("{}/execute.txt", local_baseline_dir);
                process_post_call(Some(file_name), url, payload.clone()).await?;
            }

            if x & 4u8 == 4 {
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

            if x & 8u8 == 8 {
                log::info!(
                    "[execute_baseline_flow] baseline calling llm endpoint (current state profile)"
                );
                // set the initial prompt for the llm
                let prompt = get_profile_prompt(code.clone(), ncu_report.clone()).replace("\n", "");
                // call the llm endpoint
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/llm_state_response.txt", local_baseline_dir);
                state = process_post_call(Some(file_name), url, prompt).await?;
            } else {
                state =
                    fs::read_to_string(format!("{}/llm_state_response.txt", local_baseline_dir))?;
            }

            if x & 16u8 == 16 {
                log::info!("[execute_baseline_flow] baseline calling llm endpoint (match state)");
                // set the initial prompt for the llm
                let prompt = get_state_match_prompt(state.clone());
                // call the llm endpoint
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/llm_match_state_response.txt", local_baseline_dir);
                match_state = process_post_call(Some(file_name), url, prompt).await?;
            } else {
                match_state = fs::read_to_string(format!(
                    "{}/llm_match_state_response.txt",
                    local_baseline_dir
                ))?;
            }

            if x & 32u8 == 32 {
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
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/optimization-plan.json", local_baseline_dir);
                json_plan = process_post_call(Some(file_name), url, prompt_op).await?;
            } else {
                json_plan =
                    fs::read_to_string(format!("{}/optimization-plan.json", local_baseline_dir))?;
            }

            if x & 64u8 == 64 {
                let start = Instant::now();
                log::info!("[execute_baseline_flow] executing rollout");
                let json_plan_updated = json_plan.replace("```json", "");
                let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;
                let category = Profile::get_category(state)?;
                // let mut futs = FuturesUnordered::new();
                for x in 0..parameters.max_rollout {
                    let plan = pick_weighted(plans.clone())?;
                    // Generate a shorter 8-character ID (6 bytes)
                    let short = short_id_with_bytes(6)?;
                    let trajectory_dir = format!(
                        "{}/logs/{}/{}/rl-ncu/trajectory_{}_{}",
                        parameters.working_dir,
                        parameters.llm_model,
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

                    let url = format!(
                        "{}/v1/prompt/{}",
                        parameters.llm_server_url,
                        parameters.llm_agent.to_string().to_lowercase()
                    );
                    // this will always start with step 0, then within the complex flow
                    // each new step (until max.rollout) will be added
                    let prompt_file_name = format!(
                        "{}/step_0/{}_llm_response.txt",
                        trajectory_dir, plan.technique
                    );
                    let kernel_file_name =
                        format!("{}/step_0/{}.cu", trajectory_dir, plan.technique);

                    // execute in parallel
                    // Due to rate limiting on the cerebras endpoints
                    // we can't execute theses calls in parallel
                    // commenting the code for now
                    //futs.push(extract_code_from_call(
                    //    Some(prompt_file_name),
                    //    kernel_file_name,
                    //    url,
                    //    task_prompt,
                    //));

                    // if this fails exit immediately
                    extract_code_from_call(
                        Some(prompt_file_name),
                        kernel_file_name,
                        url,
                        task_prompt,
                    )
                    .await?
                }

                // see the comment above
                // wait for the remaining to finish.
                // while let Some(response) = futs.next().await {
                //    match response {
                //        Ok(_) => log::info!(
                //            "[execute_baseline_flow] call (extract_code_from_call) succeeded"
                //        ),
                //        Err(e) => log::error!("[execute_baseline_flow] call failed {}", e),
                //    }
                //}

                let elapsed = start.elapsed();
                log::info!("[execute_baseline_flow] completed rollout in {:?}", elapsed);
            }
        }
        log::info!("[execute_baseline_flow] workflow controller completed successfully");
        Ok(())
    }

    async fn execute_agent_flow(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: work out batch processing for all level1 -> level3 kernels
        for item in parameters.workflow_batch.iter() {
            // get baseline state and elapsed_cycles
            // optimization plans were calculated in the baseline run
            // no optimization plan (json) file is found in the step_0 directories
            let baseline_dir = format!(
                "{}/logs/{}/{}/rl-ncu/baseline",
                parameters.working_dir, parameters.llm_model, item
            );
            let baseline_ncu_report = fs::read_to_string(format!("{}/profile.txt", baseline_dir))?;
            let baseline_elapsed_cycles = Profile::get_elapsed_cycles(baseline_ncu_report.clone())?;
            let trajectories_dir = format!(
                "{}/logs/{}/{}/rl-ncu",
                parameters.working_dir, parameters.llm_model, item
            );

            // display only
            let trajectories = get_trajectories(trajectories_dir.clone())?;
            log::debug!("[execute_agent_flow] trajectories {:#?}", trajectories);

            let current_trajectory = "trajectory_1_Po9t6Fh_";
            log::info!("[execute_agent_flow] trajectory   : {}", current_trajectory);
            let &mut mut fallback = &mut false;
            let mut plan_count = parameters.max_rollout - 1;

            for step in parameters.rollout_start..parameters.max_rollout {
                log::info!("[execute_agent_flow] current step : {}", step);
                let local_target_dir = format!(
                    "{}/logs/{}/{}/rl-ncu/{}/step_{}",
                    parameters.working_dir, parameters.llm_model, item, current_trajectory, step
                );
                let target_dir = local_target_dir.replace("/logs/", "/out/");
                log::info!(
                    "[execute_agent_flow] replay buffer directory : {}",
                    local_target_dir
                );

                let next_target_dir = format!(
                    "{}/logs/{}/{}/rl-ncu/{}/step_{}",
                    parameters.working_dir,
                    parameters.llm_model,
                    item,
                    current_trajectory,
                    step + 1
                );
                if step < parameters.max_rollout - 1 {
                    let res = fs::create_dir_all(next_target_dir.clone());
                    match res {
                        Ok(_) => {
                            log::info!(
                                "[execute_agent_flow] created next step directory {}",
                                next_target_dir
                            );
                        }
                        Err(e) => {
                            log::error!("[execute_agent_flow] failed to create directory {}", e);
                            fallback = true;
                            continue;
                        }
                    }
                }

                // N.B. the error handling is purposely set to "try" not break and exit the loop
                // If there is an error the objective is to continue to the next step using the step_0 kernel
                // Nested Ok() Err() checks have been avoided for legibility reasons

                let mut payload = String::new();
                let mut code = String::new();

                // compile retry is hard coded to 2
                for i in 0..2 {
                    // 1. read kernel code
                    // If for any reason the cuda kernel file cannot be read exit immediately
                    log::info!("[execute_agent_flow] compile retry loop {}", i);
                    let (cuda_file, kernel_code) =
                        find_cuda_file(local_target_dir.clone(), &mut fallback)?;
                    log::info!("[execute_agent_flow] using kernel file {}", cuda_file);
                    payload = format!(
                        r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_name": "{}" , "code": {:?} }}"##,
                        item,
                        parameters.working_dir,
                        parameters.gpu_arch,
                        target_dir,
                        cuda_file,
                        kernel_code
                    );

                    // 2. upload kernel
                    log::info!("[execute_agent_flow] uploading kernel {}", cuda_file);
                    let url = format!("{}/v1/upload", parameters.compile_server_url);
                    let upload_res = process_post_call(None, url.clone(), payload.clone()).await;
                    match upload_res {
                        Ok(_) => {
                            log::info!("[execute_agent_flow] kernel uploaded successfully");
                        }
                        Err(e) => {
                            log::error!("[execute_agent_flow] kernel upload error {}", e);
                            fallback = true;
                            continue;
                        }
                    }

                    // 3. compile kernel
                    log::info!("[execute_agent_flow] calling compile cuda kernel endpoint",);
                    let url = format!("{}/v1/compile", parameters.compile_server_url);
                    let file_name = format!("{}/compile.txt", local_target_dir);
                    let res = process_post_call(Some(file_name), url, payload.clone()).await;
                    match res {
                        Ok(_) => {
                            log::info!(
                                "[execute_agent_flow] compile kernel completed successfully"
                            );
                            code = kernel_code;
                        }
                        Err(e) => {
                            log::error!("[execute_agent_flow] compile failed {}", e);
                            fallback = true;
                            continue;
                        }
                    }
                }

                // the compile failed after 2 re-tries
                if code.is_empty() {
                    break;
                }

                // 4. execute kernel
                log::info!("[execute_agent_flow] calling execute cuda kernel endpoint");
                let url = format!("{}/v1/execute", parameters.gpu_server_url);
                let file_name = format!("{}/execute.txt", local_target_dir);
                let exec_res = process_post_call(Some(file_name), url, payload.clone()).await;
                match exec_res {
                    Ok(_) => {
                        log::info!("[execute_agent_flow] kernel execute completed successfully");
                    }
                    Err(e) => {
                        log::error!("[execute_agent_flow] kernel execute failed {}", e);
                        fallback = true;
                        continue;
                    }
                }

                // 5. profile kernel
                log::info!("[execute_agent_flow] calling profile cuda kernel endpoint");
                let url = format!("{}/v1/profile", parameters.gpu_server_url);
                let file_name = format!("{}/profile.txt", local_target_dir);
                let profile_res = process_post_call(Some(file_name), url, payload.clone()).await;
                let ncu_report = match profile_res {
                    Ok(profile) => profile,
                    Err(e) => {
                        log::error!("[execute_agent_flow] kernel profile failed {}", e);
                        fallback = true;
                        continue;
                    }
                };

                // 6. calculate elapsed cycles
                let elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;

                // 7. calculate improvement/degradation
                log::info!("[execute_agent_flow] elapsed cycles : {}", elapsed_cycles);
                let (perc, reward) =
                    Profile::calculate_improvement(baseline_elapsed_cycles, elapsed_cycles)?;
                log::info!(
                    "[execute_agent_flow] percentage improvement {}% : reward {}",
                    perc,
                    reward
                );
                let mut contents =
                    format!("baseline elapsed cycles : {}\n", baseline_elapsed_cycles);
                contents.push_str(&format!("elapsed cycles          : {}\n", elapsed_cycles));
                contents.push_str(&format!("improvement             : {}%\n", perc));
                contents.push_str(&format!("reward                  : {}\n", reward));
                fs::write(format!("{}/stats.txt", local_target_dir), contents)?;
                if perc < -100.0 {
                    log::error!("[execute_agent_flow] degradation is too severe");
                    fallback = true;
                    continue;
                }

                // 8. get new state from profile
                // get new state prompt based on the ncu_report
                // if previous step fails it will use the baseline ncu_report
                log::info!("[execute_agent_flow] calling llm (state)");
                let state_prompt =
                    get_profile_prompt(code.clone(), ncu_report.clone()).replace("\n", "");
                // call the llm endpoint
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/llm_state_response.txt", local_target_dir);
                let state_res = process_post_call(Some(file_name), url, state_prompt).await;
                let state = match state_res {
                    Ok(contents) => {
                        log::info!("[execute_agent_flow] calling llm state completed successfully");
                        contents
                    }
                    Err(e) => {
                        log::error!("[execute_agent_flow] calling llm state failed {}", e);
                        fallback = true;
                        continue;
                    }
                };

                // 9. get optimization plan
                // get the top_n matching optimizations in json format from the llm
                let avail_opt = get_available_optimizations();
                let prompt_op =
                    get_optimization_plan(plan_count, state.clone(), code.clone(), avail_opt);
                // call llm for optimization plan
                log::info!("[execute_agent_flow] calling llm (optimization plan)");
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/optimization-plan.json", local_target_dir);
                let json_plan_res = process_post_call(Some(file_name), url, prompt_op).await;
                let json_plan = match json_plan_res {
                    Ok(contents) => {
                        log::info!(
                            "[execute_agent_flow] calling llm json optimization plan completed successfully"
                        );
                        contents
                    }
                    Err(e) => {
                        log::error!(
                            "[execute_agent_flow] calling llm json optimization plan failed {}",
                            e
                        );
                        fallback = true;
                        continue;
                    }
                };
                // parse the optimization plan
                let json_plan_updated = json_plan.replace("```json", "").replace("```", "");
                let plans_res = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated);
                let plans = match plans_res {
                    Ok(vec_plans) => {
                        log::info!(
                            "[execute_agent_flow] parsing optimization plan completed successfully"
                        );
                        vec_plans
                    }
                    Err(e) => {
                        log::error!(
                            "[execute_agent_flow] parsing optimization plan failed {}",
                            e
                        );
                        fallback = true;
                        continue;
                    }
                };

                // before building the complex prompt for the next step
                // check if we have reached the max_rollout value
                if step == parameters.max_rollout - 1 {
                    log::info!(
                        "[execute_agent_flow] max_rollout reached : exiting flow gracefully"
                    );
                    break;
                }

                // 10. create complex prompt and execute for the next step
                // if this fails it due to regex (this will break our loop)
                let category = Profile::get_category(state.clone())?;
                // pick the weighted plan should not fail (unless WeightedIndex fails - severe)
                let plan = pick_weighted(plans.clone())?;
                let state_category = get_performance_state_category();
                // if file read files this should break the loop
                let combined = get_combined(state_category)?;
                let task_prompt = get_task_generate_code_prompt(
                    plan.technique.clone(),
                    category.to_owned(),
                    plan.description,
                    ncu_report.to_owned(),
                    code.to_owned(),
                    combined,
                );

                let wr_res = fs::write(
                    format!("{}/{}.prompt", next_target_dir, plan.technique),
                    task_prompt.clone(),
                );
                match wr_res {
                    Ok(_) => {
                        log::info!(
                            "[execute_agent_flow] saved {} prompt to disk ",
                            plan.technique
                        );
                    }
                    Err(e) => {
                        log::error!("[execute_agent_flow] failed to save prompt {}", e);
                        fallback = true;
                        continue;
                    }
                };

                log::info!("[execute_agent_flow] calling llm (task generate code)");
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/{}_llm_response.txt", next_target_dir, plan.technique);
                let contents_res = process_post_call(Some(file_name), url, task_prompt).await;
                match contents_res {
                    Ok(contents) => {
                        // extract code
                        let code = extract_code(contents)?;
                        let wr_res =
                            fs::write(format!("{}/{}.cu", next_target_dir, plan.technique), code);
                        match wr_res {
                            Ok(_) => {
                                log::info!(
                                    "[execute_agent_flow] saved cuda kernel {}",
                                    plan.technique
                                );
                                plan_count -= 1;
                                // plan_count starts at max_rollout -1 and then decrements by one until we reach 4
                                if plan_count <= 4 {
                                    plan_count = 4;
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "[execute_agent_flow] failed to save cuda kernel {}",
                                    e
                                );
                                fallback = true;
                            }
                        }
                    }
                    Err(e) => {
                        log::error!(
                            "[execute_agent_flow] process_post_call call generate code failed {}",
                            e
                        );
                        fallback = true;
                    }
                }
            }
            // finally find the optimal cuda kernel
            find_most_performant_kernel(trajectories_dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // this brings everything from parent's scope into this scope
    use super::*;
    use crate::config::load::WorkItem;
    use crate::{
        config::load::{ConfigInterface, ImplConfigInterface},
        utils::common::{find_most_performant_kernel, get_trajectories},
    };
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

        println!("[main] testing parameters {:#?}", parameters);

        let item = "level1/001_Square_matrix_multiplication";
        let model = "gpt-oss-120b";
        println!("[execute_baseline_flow] testing current flow control");
        let ncu_report = fs::read_to_string(format!(
            "logs/{}/{}/rl-ncu/baseline/profile.txt",
            model, item
        ))?;
        let elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;
        println!(
            "[execute_baseline_flow] testing elapsed_cycles {}",
            elapsed_cycles
        );
        let (result, reward) = Profile::calculate_improvement(elapsed_cycles, 7677773)?;
        println!(
            "[execute_baseline_flow] testing result {} : reward {}",
            result, reward
        );

        let (result, reward) = Profile::calculate_improvement(38724415, 38732739)?;
        println!(
            "[execute_baseline_flow] testing result {} : reward {}",
            result, reward
        );

        let json_plan = fs::read_to_string(format!(
            "logs/{}/{}/rl-ncu/trajectory_1_mINMOfqW/step_0/optimization-plan.json",
            model, item
        ))?;
        let state = fs::read_to_string(format!(
            "logs/{}/{}/rl-ncu/baseline/llm_state_response.txt",
            model, item
        ))?;

        let category = Profile::get_category(state)?;
        println!("[execute_baseline_flow] testing category {}", category);

        println!();

        let json_plan_updated = json_plan.replace("```json", "").replace("```", "");
        let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;

        println!();

        for _x in 0..3 {
            let pick = pick_weighted(plans.clone())?;
            println!(
                "[execute_baseline_flow] testing weighted plan {} {}",
                pick.technique, pick.relevance_score
            );
        }

        let base_dir = format!(
            "{}/logs/{}/{}/rl-ncu/trajectory_1_Po9t6Fh_/step_0",
            parameters.working_dir, model, item
        );
        let (cuda_file, cuda_kernel) = find_cuda_file(base_dir.clone(), &mut false)?;
        println!("[run] cuda kernel file {}", cuda_file);
        println!("[run] cuda kernel len {}", cuda_kernel.len());
        let updated = cuda_kernel.replace("\u{a0}", " ").replace("\u{20}", " ");

        let payload = format!(
            r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_name": "{}" , "code": {:?} }}"##,
            item, parameters.working_dir, parameters.gpu_arch, base_dir, cuda_file, updated
        );

        let work_item_res = serde_json::from_slice::<WorkItem>(&payload.as_bytes())?;
        println!("[run] work items {:?}", work_item_res);

        //let cuda_kernel = fs::read_to_string("tests/tensor_core_utilization.cu")?;
        let cuda_kernel = fs::read_to_string("tests/memory_compute_overlap.cu")?;
        let vec_lines: Vec<&str> = cuda_kernel.split("\n").collect();
        let re = Regex::new("[_]{2}global[_]{2}[\\svoid\\s]+([a-zA-Z0-9_]*)")?;
        let re_simple = Regex::new("([a-zA-Z0-9_]+)")?;
        let mut kernel_name;
        for (count, line) in vec_lines.iter().enumerate() {
            if line.contains("__global__ void") && !line.contains("__launch_bounds__") {
                for cap in re.captures_iter(line) {
                    kernel_name = cap[1].to_string();
                    println!("[run] profiling kernel {}", kernel_name);
                }
            }
            if line.contains("__global__ void __launch_bounds__")
                | line.contains("__global__ __launch_bounds__")
            {
                for cap in re_simple.captures_iter(vec_lines[count + 1]) {
                    if !cap[1].to_string().contains("void") {
                        kernel_name = cap[1].to_string();
                        println!(
                            "[run] profiling with __launch_bounds__ kernel {}",
                            kernel_name
                        );
                    }
                }
            }
        }

        let vec_trajectories = get_trajectories(format!(
            "{}/logs/{}/{}/rl-ncu",
            parameters.working_dir, model, item
        ))?;
        println!("{:?}", vec_trajectories);

        find_most_performant_kernel(format!(
            "{}/logs/{}/{}/rl-ncu",
            parameters.working_dir, model, item
        ))?;

        Ok(())
    }
}
