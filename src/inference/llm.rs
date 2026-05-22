use crate::config::load::WorkItem;
use custom_logger as log;
use orx_parallel::*;
use std::fs;
use std::process::Command;
use std::time::Instant;

pub trait LlmInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Llm {}

impl LlmInterface for Llm {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] compiling cuda-kernel");
        let start = Instant::now();

        // we can add as many concurrent compilation calls
        // but we limit this to 1
        let results: Vec<(usize, bool, String)> = (0..1usize)
            .par()
            .map(|i| {
                println!("creating job {i}");
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

                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                (i, output.status.success(), stdout)
            })
            .collect();

        let elapsed = start.elapsed();
        let failures = results.iter().filter(|(_, ok, _)| !ok).count();

        for (i, ok, out) in &results {
            let status = if *ok { "OK" } else { "FAIL" };
            println!("[{i:03}] {status} | {out}");
        }

        println!(
            "\n{} commands completed ({} failed) in {:.2?}",
            results.len(),
            failures,
            elapsed
        );

        Ok("exit => 0".to_string())
    }
}

#[allow(unused)]
fn get_api_payload(payload_path: String) -> Result<String, Box<dyn std::error::Error>> {
    let data = fs::read_to_string(payload_path)?;
    Ok(data)
}
