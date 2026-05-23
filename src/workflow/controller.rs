use crate::config::load::Parameters;
use crate::workflow::api_client::{process_get_call, process_post_call};
use custom_logger as log;

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
            let payload = format!(r##"{{ "name": "{}", "gpu_arch": "{} }}"##, item, "86");
            // first call the compile endpoint
            let mut url = format!("{}/v1/compile", parameters.compile_server_url);
            process_post_call(
                item.to_string(),
                url,
                "baseline_compile".to_string(),
                payload.clone(),
            )
            .await?;
            // call the execute endpoint
            url = format!("{}/v1/execute", parameters.gpu_server_url);
            process_post_call(
                item.to_string(),
                url,
                "baseline_execute".to_string(),
                payload.clone(),
            )
            .await?;
            // call the nvidia ncu profile endpoint
            url = format!("{}/v1/profile", parameters.gpu_server_url);
            process_post_call(
                item.to_string(),
                url,
                "baseline_profile".to_string(),
                payload,
            )
            .await?;
        }
        Ok(())
    }
}
