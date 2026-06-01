use crate::config::load::WorkItem;
use custom_logger as log;
use regex::Regex;
use std::env;
use std::fs;
use std::process::Command;
use std::time::Instant;

pub trait CompileInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
    async fn cuda_kernel(
        work_item: WorkItem,
        write: bool,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Compile {}

impl CompileInterface for Compile {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] compiling cuda-kernel");
        let start = Instant::now();

        // create output directory
        fs::create_dir_all(format!(
            "out/{}/{}/build",
            work_item.name, work_item.target_dir
        ))?;

        // create the cuda_model.cuh file
        // we use regex to extract it
        let content =
            fs::read_to_string(format!("kernelbench-cuda/{}/driver.cpp", work_item.name))?;
        let re = Regex::new("(void launch_gpu_implementation\\([\\n\\s,\\/\\(\\)a-zA-Z0-9*_]*;)")?;
        for cap in re.captures_iter(&content) {
            fs::write(
                format!(
                    "out/{}/{}/cuda_model.cuh",
                    work_item.name, work_item.target_dir
                ),
                cap[1].as_bytes(),
            )?;
        }

        // copy driver.cpp
        fs::copy(
            format!("kernelbench-cuda/{}/driver.cpp", work_item.name),
            format!("out/{}/{}/main.cpp", work_item.name, work_item.target_dir),
        )?;

        // read init.cu and insert #include cuda_model.cuh
        // then save to the out directory
        let mut kernel =
            fs::read_to_string(format!("kernelbench-cuda/{}/init.cu", work_item.name))?;
        let res = kernel.rfind("#include").unwrap_or(0);
        let insert = "#include \"cuda_model.cuh\"\n";
        kernel.insert_str(res, insert);
        fs::write(
            format!(
                "out/{}/{}/cuda_model.cu",
                work_item.name, work_item.target_dir
            ),
            &kernel,
        )?;

        // copy CMakeLists.txt
        fs::copy(
            "kernelbench-cuda/CMakeLists.txt",
            format!(
                "out/{}/{}/CMakeLists.txt",
                work_item.name, work_item.target_dir
            ),
        )?;

        // step 1 - call cmake to build MakeFile
        env::set_current_dir(format!(
            "out/{}/{}/build",
            work_item.name, work_item.target_dir
        ))?;
        let output = Command::new("cmake")
            .arg("-DCMAKE_PREFIX_PATH=/usr/local/libtorch")
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg("..")
            .arg(format!("-DGPU_ARCH_VERSION={}", work_item.gpu_arch))
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let elapsed = start.elapsed();
        // preserve output
        println!("{}", stdout);
        log::info!("[run] compile : completed task in {:?}", elapsed);

        if !output.status.success() {
            // return the first failure (fail fast)
            return Err(Box::from(stderr));
        }

        // step 2 - build binary
        let output = Command::new("cmake")
            .arg("--build")
            .arg(".")
            .arg("--config")
            .arg("Release")
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let elapsed = start.elapsed();
        // preserve output
        println!("{}", stdout);
        log::info!("[run] compile build : completed task in {:?}", elapsed);

        if !output.status.success() {
            // return the first failure (fail fast)
            return Err(Box::from(stderr));
        }

        // restore working dir
        env::set_current_dir(work_item.working_dir)?;
        Ok(stdout)
    }

    async fn cuda_kernel(
        work_item: WorkItem,
        write: bool,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut kernel = String::new();
        let dir = format!(
            "{}/out/{}/{}",
            work_item.working_dir, work_item.name, work_item.target_dir
        );
        let file = format!("{}/cuda_mode.cu", dir);
        if write {
            fs::create_dir_all(dir)?;
            fs::write(file, work_item.code)?;
        } else {
            kernel = fs::read_to_string(&file)?;
        }
        Ok(kernel)
    }
}
