use crate::config::load::WorkItem;
use custom_logger as log;
use regex::Regex;
use std::env;
use std::fs;
use std::process::Command;
use std::time::Instant;

pub trait ProfileInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
    fn get_elapsed_cycles(ncu_report: String) -> Result<i64, Box<dyn std::error::Error>>;
    fn calculate_improvement(
        baseline: i64,
        current: i64,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>>;
    fn get_category(state_report: String) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Profile {}

impl ProfileInterface for Profile {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let mut profile_buffer = String::new();

        // restore working dir
        env::set_current_dir(work_item.working_dir)?;

        // ensure we read and set LD_LIBARAY_PATH envar
        let ld_lib = env::var("LD_LIBRARY_PATH")?;

        // get the kernel name
        let kernel_file = match work_item.kernel_name {
            Some(name) => format!("{}/{}", work_item.target_dir, name),
            None => format!("kernelbench-cuda/{}/init.cu", work_item.name),
        };
        let kernel = fs::read_to_string(kernel_file)?;
        // handle multiple kernel names
        let kernel_names = extract_kernel_name(kernel)?;
        log::debug!("[run] profile kernel_names {:#?}", kernel_names);
        // for profiling we set the current working directory
        env::set_current_dir(format!("{}/build", work_item.target_dir))?;

        for name in kernel_names.iter() {
            log::info!("[run] profiling kernel {}", name);
            let output = Command::new("sudo")
                .arg(format!("LD_LIBRARY_PATH={}", ld_lib))
                .arg("ncu")
                .arg("--set")
                .arg("full")
                .arg("-k")
                .arg(name.clone())
                .arg("-o")
                .arg(format!("profile-{}", name))
                .arg("-f")
                .arg("main")
                .output()?;

            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let elapsed = start.elapsed();
            log::info!("[run] profile : completed task in {:?}", elapsed);

            if !output.status.success() {
                return Err(Box::from(stderr.to_string()));
            }

            // preserve output
            println!("{}", stdout);

            let output = Command::new("sudo")
                .arg(format!("LD_LIBRARY_PATH={}", ld_lib))
                .arg("ncu")
                .arg("--import")
                .arg(format!("profile-{}.ncu-rep", name))
                .arg("--page")
                .arg("details")
                .output()?;

            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let elapsed = start.elapsed();
            log::info!("[run] profile convert : completed task in {:?}", elapsed);

            if !output.status.success() {
                return Err(Box::from(stderr.to_string()));
            }

            // write the final report to disk
            fs::write(format!("{}.profile", name), stdout.clone())?;
            profile_buffer.push('\n');
            profile_buffer.push_str(&stdout);
        }

        Ok(profile_buffer)
    }

    fn get_elapsed_cycles(ncu_report: String) -> Result<i64, Box<dyn std::error::Error>> {
        let re = Regex::new("[\\s]{4}Elapsed\\sCycles[\\s]+cycle[\\s]+([0-9,]*)")?;
        let mut elapsed_cycles = 0;
        for cap in re.captures_iter(&ncu_report) {
            elapsed_cycles = cap[1].to_string().replace(",", "").parse::<i64>()?;
            log::trace!("[get_elapsed_cycles] {}", elapsed_cycles);
        }
        Ok(elapsed_cycles)
    }

    fn get_category(state_report: String) -> Result<String, Box<dyn std::error::Error>> {
        let re = Regex::new("[\\*]{2}PRIMARY_BOTTLENECK:[\\*]{2}[\\s]*([`a-zA-Z_]*)")?;
        let mut category = String::new();
        for cap in re.captures_iter(&state_report) {
            category = cap[1].to_string().replace("`", "").replace("_bound", "");
            log::trace!("[get_category] {}", category);
        }
        Ok(category)
    }

    fn calculate_improvement(
        baseline: i64,
        current: i64,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut reward = (baseline - current) as f64 / baseline as f64;
        if reward < 0.0 {
            // add penalty for being worse
            reward += -0.5
        }
        let result = reward * 100.0;
        Ok((result, reward))
    }
}

fn extract_kernel_name(cuda_kernel: String) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut vec_res = vec![];
    let vec_lines: Vec<&str> = cuda_kernel.split("\n").collect();
    let re = Regex::new("[_]{2}global[_]{2}[\\svoid\\s]+([a-zA-Z0-9_]*)")?;
    let re_simple = Regex::new("([a-zA-Z0-9_]+)")?;
    for (count, line) in vec_lines.iter().enumerate() {
        if line.contains("__global__ void") && !line.contains("__launch_bounds__") {
            for cap in re.captures_iter(line) {
                vec_res.push(cap[1].to_string());
            }
        }
        if line.contains("__global__ void __launch_bounds__")
            | line.contains("__global__ __launch_bounds__")
        {
            for cap in re_simple.captures_iter(vec_lines[count + 1]) {
                if !cap[1].to_string().contains("void") {
                    vec_res.push(cap[1].to_string());
                }
            }
        }
    }
    Ok(vec_res)
}
