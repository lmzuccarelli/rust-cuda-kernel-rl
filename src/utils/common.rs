use crate::workflow::api_client::process_post_call;
use crate::workflow::controller::OptimizationPlan;
use custom_logger as log;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use std::fs;
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
                                    let vec_parts = file.split("step_").collect::<Vec<&str>>();
                                    let target_dir = vec_parts[0].replace("logs", "out");
                                    let file_name = format!(
                                        "step_{}",
                                        vec_parts[1].replace("_llm_response.txt", ".cu")
                                    );
                                    log::info!("target_dir : {}", target_dir);
                                    log::info!("file_name  : {}", file_name);
                                    // write to local disk
                                    let res = fs::write(
                                        format!("{}/{}", vec_parts[0], file_name),
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

pub async fn walk_trajectories(base_dir: String) -> Result<(), Box<dyn std::error::Error>> {
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_file() {
                    let file = obj.path().to_string_lossy();
                    if file.contains(".cu") && file.contains("trajectory_") {
                        let contents_res = fs::read_to_string(file.to_string());
                        match contents_res {
                            Ok(_) => {
                                let parts = file.split("/").collect::<Vec<&str>>();
                                let target_dir =
                                    format!("/{}", parts[1..parts.len() - 1].join("/"));
                                log::info!(
                                    "[walk_trajectories] target_dir {}",
                                    target_dir.replace("/logs/", "/out/")
                                );
                            }
                            Err(e) => {
                                log::error!("[walk_trajectories] error reading {}", e);
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
