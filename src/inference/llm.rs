use crate::config::load::WorkItem;
use custom_logger as log;
use std::process::Command;
use std::time::Instant;

pub trait LlmInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Llm {}

impl LlmInterface for Llm {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing llm inference endpoint");
        let start = Instant::now();

        let output = Command::new("claude")
            .args(vec![
                "-p",
                &work_item
                    .prompt
                    .clone()
                    .unwrap_or("default prompt".to_string()),
            ])
            .output()
            .expect("failed to spawn shell");

        let elapsed = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        // preserve output
        println!("{}", stdout);
        log::info!("completed task in {:?}", elapsed);

        if !output.status.success() {
            // return the first failure (fail fast)
            return Err(Box::from(stderr));
        }

        Ok("exit => 0".to_string())
    }
}
