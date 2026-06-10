use crate::inference::llm::{Llm, LlmInterface};
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
            x if x.contains("/v1/prompt") => {
                let data = req.into_body().collect().await?.to_bytes();
                let prompt_res = String::from_utf8(data.to_vec());
                match prompt_res {
                    Ok(prompt) => {
                        let result = Llm::run(prompt).await;
                        match result {
                            Ok(content) => {
                                *response.body_mut() = Full::from(content);
                            }
                            Err(err) => {
                                log::error!("[endpoints] llm prompt request error {}", err);
                                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                                *response.body_mut() = Full::from(format!(
                                    "[endpoints] llm prompt request error {}\n",
                                    err
                                ));
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("[endpoints] prompt request parsing body error {}", err);
                        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                        *response.body_mut() = Full::from(format!(
                            "[endpoints] prompt request parsing body error {}\n",
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
                    r##"{{ "status": "ok", "appplication": "{}", "service": "llm-inference", "version": "{}" }}"##,
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
