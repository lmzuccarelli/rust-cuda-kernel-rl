use crate::config::load::WorkItem;
use custom_logger as log;
use regex::Regex;
use std::env;
use std::fs;
use std::process::Command;
use std::time::Instant;

pub trait ProfileInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
    fn get_elapsed_cycles(ncu_report: String) -> Result<u64, Box<dyn std::error::Error>>;
    fn calculate_improvement(
        baseline: u64,
        current: u64,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>>;
}

pub struct Profile {}

impl ProfileInterface for Profile {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] profiling cuda-kernel");
        let start = Instant::now();

        // ensure we read and set LD_LIBARAY_PATH envar
        let ld_lib = env::var("LD_LIBRARY_PATH")?;

        // get the kernel name
        let kernel = fs::read_to_string(format!("out/{}/cuda_model.cu", work_item.name))?;
        let re = Regex::new("[_]{2}global[_]{2}\\svoid\\s([a-zA-Z0-9_]*)")?;
        let mut kernel_name = String::new();
        for cap in re.captures_iter(&kernel) {
            kernel_name = cap[1].to_string();
            log::info!("[run] profile kernel name {}", kernel_name);
        }
        if kernel_name.is_empty() {
            return Err(Box::from("[run] profile could not find kernel name"));
        }

        // for profiling we set the current working directory
        env::set_current_dir(format!("out/{}/build", work_item.name))?;
        let output = Command::new("sudo")
            .arg(format!("LD_LIBRARY_PATH={}", ld_lib))
            .arg("ncu")
            .arg("--set")
            .arg("full")
            .arg("-k")
            .arg(kernel_name)
            .arg("-o")
            .arg("profile")
            .arg("-f")
            .arg("main")
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let elapsed = start.elapsed();
        log::info!("completed task in {:?}", elapsed);

        if !output.status.success() {
            return Err(Box::from(stderr.to_string()));
        }

        // preserve output
        println!("{}", stdout);

        let output = Command::new("sudo")
            .arg(format!("LD_LIBRARY_PATH={}", ld_lib))
            .arg("ncu")
            .arg("--import")
            .arg("profile.ncu-rep")
            .arg("--page")
            .arg("details")
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let elapsed = start.elapsed();
        log::info!("completed task in {:?}", elapsed);

        if !output.status.success() {
            return Err(Box::from(stderr.to_string()));
        }

        // write the final report to disk
        fs::write("output.profile", stdout.clone())?;

        // restore working dir
        env::set_current_dir(work_item.working_dir)?;
        Ok(stdout)
    }

    fn get_elapsed_cycles(ncu_report: String) -> Result<u64, Box<dyn std::error::Error>> {
        let re = Regex::new("[\\s]{4}Elapsed\\sCycles[\\s]+cycle[\\s]+([0-9,]*)")?;
        let mut elapsed_cycles = 0;
        for cap in re.captures_iter(&ncu_report) {
            elapsed_cycles = cap[1].to_string().replace(",", "").parse::<u64>()?;
            log::trace!("[get_initial_elapsed_cycles] {}", elapsed_cycles);
        }
        Ok(elapsed_cycles)
    }

    fn calculate_improvement(
        baseline: u64,
        current: u64,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let reward = (baseline - current) as f64 / baseline as f64;
        let result = reward * 100.0;
        Ok((result, reward))
    }
}
