use crate::config::load::WorkItem;
use crate::kernel::compile::{Compile, CompileInterface};
use custom_logger as log;
use http::{Method, Request, Response, StatusCode};
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};

pub async fn endpoints(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let mut response = Response::new(Full::default());
    let request = req.uri().path();
    log::debug!("{}", request);
    match *req.method() {
        Method::POST => match request {
            x if x.contains("/v1/compile") => {
                let data = req.into_body().collect().await?.to_bytes();
                let work_item_res = serde_json::from_slice(&data);
                match work_item_res {
                    Ok(work_item) => {
                        let compile_res = Compile::run(work_item).await;
                        match compile_res {
                            Ok(content) => {
                                *response.body_mut() = Full::from(content);
                            }
                            Err(err) => {
                                log::error!("[endpoints] compiling cuda kernel error {}", err);
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] compiling cuda kernel error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!(
                            "[endpoints] compiling cuda kernel parsing body error {}",
                            err
                        );
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] compiling cuda kernel parsing body error {}\n",
                            err
                        ));
                    }
                }
            }
            x if x.contains("/v1/cuda-kernel") => {
                let data = req.into_body().collect().await?.to_bytes();
                let work_item_res = serde_json::from_slice(&data);
                match work_item_res {
                    Ok(work_item) => {
                        let content_res = Compile::cuda_kernel_rw(work_item, false).await;
                        match content_res {
                            Ok(content) => {
                                *response.body_mut() = Full::from(content);
                            }
                            Err(err) => {
                                log::error!("[endpoints] read/write cuda kernel error {}", err);
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] read/write cuda kernel error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!(
                            "[endpoints] read/write cuda kernel parsing body error {}",
                            err
                        );
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] read/write cuda kernel parsing body error {}\n",
                            err
                        ));
                    }
                }
            }
            x if x.contains("/v1/upload") => {
                let data = req.into_body().collect().await?.to_bytes();
                let work_item_res = serde_json::from_slice::<WorkItem>(&data);
                match work_item_res {
                    Ok(work_item) => {
                        let content_res = Compile::cuda_kernel_rw(work_item.clone(), true).await;
                        match content_res {
                            Ok(_) => {
                                *response.status_mut() = StatusCode::OK;
                                *response.body_mut() = Full::from("ok");
                            }
                            Err(err) => {
                                log::error!(
                                    "[endpoints] uploading cuda kernel {} : {:?}",
                                    err,
                                    work_item
                                );
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] uploading cuda kernel error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!(
                            "[endpoints] uploading cuda kernel parsing body error {}",
                            err
                        );
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] uploading cuda kernel parsing body error {}\n",
                            err
                        ));
                    }
                }
            }

            &_ => {}
        },
        Method::GET => match request {
            x if x.contains("/v1/health") => {
                let mut content = format!(
                    r##"{{ "status": "ok", "appplication": "{}", "service": "compiler", "version": "{}" }}"##,
                    env!("CARGO_PKG_NAME"),
                    env!("CARGO_PKG_VERSION"),
                );
                content.push('\n');
                *response.body_mut() = Full::from(content);
            }
            &_ => {}
        },
        _ => {
            log::error!("[endpoints] method/endpoint not implemented");
            *response.body_mut() = Full::from("[endpoints] method/endpoint not implmented\n");
            *response.status_mut() = StatusCode::NOT_FOUND;
        }
    };
    Ok(response)
}
