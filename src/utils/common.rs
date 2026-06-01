use crate::workflow::api_client::process_post_call;
use custom_logger as log;
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
    item: String,
    working_dir: String,
    url: String,
    gpu_arch: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = format!("logs/{}/rl-ncu", item);
    for e in WalkDir::new(dir) {
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
                                    let f = file.clone();
                                    let vec_file = f.split("_llm_response").collect::<Vec<&str>>();
                                    log::info!("file      : {}", f);
                                    log::info!("split     : {}", vec_file[0]);
                                    log::info!("file name : {:?}", obj.path().file_name());
                                    let target_dir = vec_file[0].replace("log", "out");
                                    // write to local disk
                                    let _ = fs::write(format!("{}.cu", vec_file[0]), code.clone());
                                    log::info!(
                                        "[extract_code_all] code extracted to {}.cu",
                                        vec_file[0]
                                    );
                                    // prepare for upload
                                    let payload = format!(
                                        r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}", "code": "{:?}" , "target_dir": "{}" }}"##,
                                        "cuda_model.cu", working_dir, gpu_arch, code, target_dir
                                    );
                                    // log::debug!("{}", payload);
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
