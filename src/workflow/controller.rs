use crate::config::load::Parameters;
use crate::inference::prompts::{get_profile_prompt, get_state_match_prompt};
use crate::workflow::api_client::{process_get_call, process_post_call};
use custom_logger as log;
use std::fs;

pub trait ControllerInterface {
    async fn get_health(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
    async fn get_baseline(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct Controller {}

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

    async fn get_baseline(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        for item in parameters.workflow_batch.iter() {
            // ensure logs directory is created for each cuda-kernel
            fs::create_dir_all(format!("logs/{}", item))?;

            // TODO: method to extract gpu arch sm
            // hardcoded for now
            let payload = format!(
                r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" }}"##,
                item, parameters.working_dir, "86"
            );
            let mut ncu_report = String::new();
            let mut state = String::new();

            // download cuda kernel (init.cu)
            log::info!("downloading cuda kernel");
            let url = format!("{}/v1/cuda-kernel", parameters.compile_server_url);
            let code = process_post_call(
                item.to_string(),
                url,
                "init.cu".to_string(),
                payload.clone(),
            )
            .await?;

            match parameters.flow_control {
                x if (x & 1u8) == 1 => {
                    // first call the compile endpoint
                    log::info!("calling compile cuda kernel endpoint");
                    let url = format!("{}/v1/compile", parameters.compile_server_url);
                    process_post_call(
                        item.to_string(),
                        url,
                        "baseline_compile.txt".to_string(),
                        payload.clone(),
                    )
                    .await?;
                }
                x if (x & 2u8) == 2 => {
                    // call the execute endpoint
                    log::info!("calling execute cuda kernel endpoint");
                    let url = format!("{}/v1/execute", parameters.gpu_server_url);
                    process_post_call(
                        item.to_string(),
                        url,
                        "baseline_execute.txt".to_string(),
                        payload.clone(),
                    )
                    .await?;
                }
                x if (x & 4u8) == 4 => {
                    // call the nvidia ncu profile endpoint
                    log::info!("calling profile cuda kernel endpoint");
                    let url = format!("{}/v1/profile", parameters.gpu_server_url);
                    ncu_report = process_post_call(
                        item.to_string(),
                        url,
                        "baseline_profile.txt".to_string(),
                        payload,
                    )
                    .await?;
                }
                x if (x & 8u8) == 8 => {
                    log::info!("calling llm endpoint (baseline profile)");
                    // if flow control is not set for cuda profile try read the existing file
                    if (x & 4u8) == 0 {
                        ncu_report = fs::read_to_string(format!(
                            "logs/{}/baseline_profile.txt",
                            parameters.name
                        ))?;
                    }
                    // set the initial prompt for the llm
                    let prompt = get_profile_prompt(code, ncu_report).replace("\n", "");
                    // call the llm endpoint
                    let url = format!("{}/v1/prompt", parameters.llm_server_url);
                    state = process_post_call(
                        item.to_string(),
                        url,
                        "baseline_llm_response.txt".to_string(),
                        prompt,
                    )
                    .await?;
                }
                x if (x & 16u8) == 16 => {
                    log::info!("calling llm endpoint (baseline state match)");
                    // if flow control is not set for cuda profile try read the existing file
                    if (x & 8u8) == 0 {
                        state =
                            fs::read_to_string(format!("logs/{}/baseline_llm_response.txt", item))?;
                    }
                    // set the initial prompt for the llm
                    let prompt = get_state_match_prompt(state).replace("\n", "");
                    // call the llm endpoint
                    let url = format!("{}/v1/prompt", parameters.llm_server_url);
                    process_post_call(
                        item.to_string(),
                        url,
                        "baseline_llm_match_response.txt".to_string(),
                        prompt,
                    )
                    .await?;
                }

                _ => {}
            }
        }
        log::info!("workflow controller completed successfully");
        Ok(())
    }
}
