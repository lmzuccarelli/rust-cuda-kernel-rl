use serde_derive::{Deserialize, Serialize};
use std::fs::File;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Parameters {
    pub name: String,
    pub log_level: String,
    pub gpu_server_url: String,
    pub compile_server_url: String,
    pub llm_server_url: String,
    pub workflow_batch: Vec<String>,
    pub working_dir: String,
    pub llm_model: String,
    pub token_file: Option<String>,
    pub openapi_url: Option<String>,
    pub gpu_arch: u8,
    pub max_rollout: u8,
    pub rollout_start: u8,
    pub flow_control: u8,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorkItem {
    pub name: String,
    pub gpu_arch: String,
    pub prompt: Option<String>,
    pub target_dir: String,
    pub working_dir: String,
    pub kernel_name: Option<String>,
    pub code: Option<String>,
}

pub trait ConfigInterface {
    fn read(&self, dir: String) -> Result<Parameters, Box<dyn std::error::Error>>;
}

#[derive(Debug, Clone)]
pub struct ImplConfigInterface {}

impl ConfigInterface for ImplConfigInterface {
    fn read(&self, name: String) -> Result<Parameters, Box<dyn std::error::Error>> {
        let json_data = File::open(&name)?;
        let params = serde_json::from_reader(json_data)?;
        Ok(params)
    }
}
