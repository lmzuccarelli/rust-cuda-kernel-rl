use crate::workflow::api_client::process_post_call;
use crate::workflow::controller::OptimizationPlan;
use custom_logger as log;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use regex::Regex;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;
use walkdir::WalkDir;

// common helper functions

pub fn extract_code(input: String) -> Result<String, Box<dyn std::error::Error>> {
    let start = input.find("```cpp").unwrap_or(0);
    let start = start + 7;
    let end = input[start..].find("```").unwrap_or(0);
    let end = start + end;
    let result = &input[start..end];
    Ok(result.to_string())
}

#[allow(unused)]
pub async fn extract_code_all(
    base_dir: String,
    working_dir: String,
    url: String,
    gpu_arch: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_file() {
                    let file = obj.path().to_string_lossy();
                    if file.contains("_llm_response.txt") {
                        let contents_res = fs::read_to_string(file.to_string());
                        match contents_res {
                            Ok(contents) => {
                                let code = extract_code(contents)?;
                                if !code.is_empty() {
                                    let file_name = obj
                                        .file_name()
                                        .to_string_lossy()
                                        .replace("logs", "out")
                                        .replace("_llm_response.txt", ".cu");
                                    log::info!("file_name  : {}", file_name);
                                    let target_dir = obj
                                        .path()
                                        .parent()
                                        .unwrap_or(Path::new(""))
                                        .to_string_lossy();
                                    // write to local disk
                                    let res = fs::write(
                                        format!("{}/{}", target_dir, file_name),
                                        code.clone(),
                                    );
                                    match res {
                                        Ok(_) => log::debug!(
                                            "[extract_code_all] file {} saved to local disk",
                                            file_name
                                        ),
                                        Err(e) => log::error!("[extract_code_all] {}", e),
                                    }
                                    // prepare for upload
                                    let payload = format!(
                                        r##"{{ "name": "{}", "gpu_arch": "{}", "code": {:?} , "target_dir": "{}", "working_dir": "{}" }}"##,
                                        file_name, gpu_arch, code, target_dir, working_dir
                                    );
                                    process_post_call(None, url.clone(), payload).await?;
                                } else {
                                    log::warn!("[extract_code_all] code not found {}", file);
                                }
                            }
                            Err(e) => {
                                log::error!("[extract_code_all] error reading {}", e);
                            }
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("[extract_code_all] error : {}", e);
            }
        }
    }
    Ok(())
}

pub fn find_most_performant_kernel(base_dir: String) -> Result<(), Box<dyn std::error::Error>> {
    let re = Regex::new("reward[\\s]*:\\s([0-9\\.-]+)")?;
    let mut max_reward = 0.0;
    let mut kernel_path = String::new();
    let mut reward_values = String::new();
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_file() {
                    let file = obj.path().to_string_lossy();
                    if file.contains("stats.txt") {
                        let contents_res = fs::read_to_string(file.to_string());
                        match contents_res {
                            Ok(contents) => {
                                for cap in re.captures_iter(&contents) {
                                    let reward = cap[1].to_string().parse::<f64>()?;
                                    if reward > -0.6 {
                                        reward_values.push_str(&format!("{},", reward));
                                    }
                                    if reward > max_reward {
                                        max_reward = reward;
                                        kernel_path = file.to_string();
                                    }
                                    log::debug!("[find_most_performant_kernel] reward {}", reward);
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "[find_most_performant_kernel] error reading stats file{}",
                                    e
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("[find_most_performant_kernel] error : {}", e);
            }
        }
    }
    let vec_parts: Vec<&str> = kernel_path.split("/").collect();
    let updated_path = vec_parts[..vec_parts.len() - 1].join("/").to_string();
    let (kernel_name, kernel_contents) =
        find_cuda_file(updated_path.clone(), "".to_string(), &mut false)?;
    let result_msg = format!(
        "[find_most_performant_kernel] found kernel {} in path {} with reward {}",
        kernel_name, updated_path, max_reward
    );
    log::info!("{result_msg}");
    let write_dir: Vec<&str> = updated_path.split("rl-ncu").collect();
    fs::write(
        format!("{}/rl-ncu/final_rl_cuda_perf.cu", write_dir[0]),
        kernel_contents,
    )?;
    fs::write(
        format!("{}/rl-ncu/rewards.csv", write_dir[0]),
        // exclude trailing ","
        &reward_values[0..reward_values.len() - 1],
    )?;
    fs::write(format!("{}/rl-ncu/results.txt", write_dir[0]), result_msg)?;
    Ok(())
}

pub fn pick_weighted(
    mut plans: Vec<OptimizationPlan>,
    exclude: Vec<String>,
) -> Result<OptimizationPlan, Box<dyn std::error::Error>> {
    // TODO: This was set due to high reward values
    // it could change from kernel to kernel
    let mut pick = OptimizationPlan {
        technique: "memory_coalescing_optimization".to_string(),
        relevance_score: 0.9,
        description: "Align access patterns to cache lines".to_string(),
    };
    // first exclude plans that we know don't work
    plans.retain(|x| !exclude.contains(&x.technique));

    let weights = plans
        .clone()
        .iter()
        .map(|x| x.relevance_score.powi(3))
        .collect::<Vec<f32>>();
    if weights.len() > 0 {
        log::debug!("[pick_weighted] {:?}", weights);
        let dist = WeightedIndex::new(weights)?;
        let mut rng = rand::rng();
        let index = dist.sample(&mut rng);
        log::debug!("[pick_weighted] using index {}", index);
        //pick = plans[index].clone();
        pick = plans[0].clone();
    } else {
        log::debug!("[pick_weighted] using default {}", pick.technique);
    }
    Ok(pick)
}

pub fn find_cuda_file(
    dir: String,
    fallback_kernel: String,
    fallback: &mut bool,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    log::trace!("[find_cuda_file] fallback {}", *fallback);
    log::trace!("[find_cuda_file] using directory {}", dir);
    let mut cuda_kernel = String::new();
    let mut cuda_file = String::new();
    if !*fallback {
        let files = fs::read_dir(dir.clone())?;
        for f in files {
            match f {
                Ok(name) => {
                    let cf = name.file_name().to_string_lossy().to_string();
                    if cf.contains(".cu") {
                        // find the first kernel file in the directory
                        cuda_file = cf.clone();
                        let contents = fs::read_to_string(format!("{}/{}", dir, cf))?;
                        cuda_kernel = contents.chars().filter(|c| c.is_ascii()).collect();
                        break;
                    }
                }
                Err(e) => {
                    return Err(Box::from(format!("[find_cuda_file] error {}", e)));
                }
            }
        }
    }
    if *fallback | cuda_kernel.is_empty() {
        if fallback_kernel.is_empty() {
            log::info!("[find_cuda_file] fallback to use baseline init.cu");
            let baseline_fallback_path = dir.split("trajectory_").next().unwrap_or("");
            let contents =
                fs::read_to_string(format!("{}/baseline/init.cu", baseline_fallback_path))?;
            fs::copy(
                format!("{}/baseline/init.cu", baseline_fallback_path),
                format!("{}/init.cu", dir),
            )?;
            cuda_kernel = contents.chars().filter(|c| c.is_ascii()).collect();
            cuda_file = "init.cu".to_string();
        } else {
            let contents = fs::read_to_string(&fallback_kernel)?;
            cuda_file = fallback_kernel.split("/").last().unwrap_or("").to_string();
            cuda_kernel = contents.chars().filter(|c| c.is_ascii()).collect();
        }
        *fallback = false;
    }
    Ok((cuda_file, cuda_kernel))
}

pub async fn extract_code_from_call(
    prompt_file_name: Option<String>,
    kernel_file_name: String,
    url: String,
    data: String,
) -> Result<(), Box<dyn std::error::Error>> {
    log::debug!(
        "[extract_code_from_call] prompt file name {:?}",
        prompt_file_name
    );
    log::debug!(
        "[extract_code_from_call] kernel file name {}",
        kernel_file_name
    );
    let contents = process_post_call(prompt_file_name, url, data).await?;
    let code = extract_code(contents)?;
    fs::write(kernel_file_name, code)?;
    log::info!("[extract_code_from_call] delaying thread (limit requests)");
    thread::sleep(Duration::from_secs(5));
    Ok(())
}

pub fn get_trajectories(
    base_dir: String,
    exclude_trajectories: Vec<String>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut vec_trajectories = vec![];
    let re = Regex::new("(trajectory_[0-9]+_[0-9a-zA-Z_-]*)$")?;
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_dir() {
                    let file = obj.path().to_string_lossy();
                    for cap in re.captures_iter(&file) {
                        let trajectory = cap[1].to_string();
                        if !exclude_trajectories.contains(&trajectory) {
                            vec_trajectories.push(cap[1].to_string());
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("[get_trajectories] error : {}", e);
            }
        }
    }
    Ok(vec_trajectories)
}
