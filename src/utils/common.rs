use crate::workflow::api_client::process_post_call;
use crate::workflow::controller::OptimizationPlan;
use custom_logger as log;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use regex::Regex;
use std::fs;
use std::path::Path;
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
    let re = Regex::new("reward[\\s]*:\\s([0-9\\.]+)")?;
    let mut max_reward = 0.0;
    let mut kernel_path = String::new();
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
                                    if reward > max_reward {
                                        max_reward = reward;
                                        kernel_path = file.to_string();
                                    }
                                    log::info!("[find_most_performant_kernel] reward {}", reward);
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
                log::error!("[walk_trajectories] error : {}", e);
            }
        }
    }
    let vec_parts: Vec<&str> = kernel_path.split("/").collect();
    let updated_path = format!("{}", vec_parts[..vec_parts.len() - 1].join("/"));
    let (kernel_name, kernel_contents) = find_cuda_file(updated_path.clone(), &mut false)?;
    log::info!(
        "find_most_performant_kernel] found kernel {} in path {} with reward {}",
        kernel_name,
        updated_path,
        max_reward
    );
    let write_dir: Vec<&str> = updated_path.split("rl-ncu").collect();
    fs::write(
        format!("{}/rl-ncu/final_rl_cuda_perf.cu", write_dir[0]),
        kernel_contents,
    )?;
    Ok(())
}

pub fn pick_weighted(
    plans: Vec<OptimizationPlan>,
) -> Result<OptimizationPlan, Box<dyn std::error::Error>> {
    let weights = plans
        .clone()
        .iter()
        .map(|x| x.relevance_score.powi(3))
        .collect::<Vec<f32>>();
    log::debug!("[pick_weighted] {:?}", weights);
    let dist = WeightedIndex::new(weights)?;
    let mut rng = rand::rng();
    let pick = plans[dist.sample(&mut rng)].clone();
    Ok(pick)
}

pub fn find_cuda_file(
    dir: String,
    fallback: &mut bool,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let files = fs::read_dir(dir.clone())?;
    log::trace!("[find_cuda_file] using directory {}", dir);
    let mut cuda_kernel = String::new();
    let mut cuda_file = String::new();
    if !*fallback {
        for f in files {
            match f {
                Ok(name) => {
                    let f = name.file_name().to_string_lossy().to_string();
                    if f.contains(".cu") {
                        // find the first kernel file in the directory
                        cuda_file = f.clone();
                        cuda_kernel = fs::read_to_string(format!("{}/{}", dir, f))?;
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
        let fallback_path = dir.split("trajectory_").next().unwrap_or("baseline");
        // use the baseline, I found that by falling back to the previous
        // step perpetuates errors and they just repeat going forward
        let fallback_kernel = format!("{}baseline/init.cu", fallback_path);
        log::info!("[find_cuda_file] using fallback kernel {}", fallback_kernel);
        cuda_kernel = fs::read_to_string(&fallback_kernel)?;
        cuda_file = "init.cu".to_string();
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
    let contents = process_post_call(prompt_file_name, url, data).await?;
    let code = extract_code(contents)?;
    fs::write(kernel_file_name, code)?;
    Ok(())
}

#[allow(unused)]
pub fn get_trajectories(base_dir: String) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut vec_trajectories = vec![];
    let re = Regex::new("(trajectory_[0-9]+_[0-9a-zA-Z_-]*)$")?;
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_dir() {
                    let file = obj.path().to_string_lossy();
                    for cap in re.captures_iter(&file) {
                        vec_trajectories.push(cap[1].to_string());
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
