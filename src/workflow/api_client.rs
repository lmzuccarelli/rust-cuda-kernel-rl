use custom_logger as log;
use hyper::StatusCode;
use reqwest::Client;
use std::fs;

// this is a complex post as it will call the endpoint
pub async fn process_post_call(
    name: String,
    url: String,
    title: String,
    data: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;
    let client_response = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(data)
        .send()
        .await?;

    let status = client_response.status();
    let response = client_response.bytes().await?;

    let res = match status {
        StatusCode::OK => {
            // only if we have success can we then save the document
            let doc_content = String::from_utf8(response.to_vec())?;
            fs::write(format!("logs/{}/{}", name, title), doc_content.clone())?;
            doc_content
        }
        _ => {
            return Err(Box::from(format!("post request failed {}", status)));
        }
    };
    log::trace!("[process_post_call] response {:?}", res);
    Ok(res)
}

pub async fn process_get_call(url: String) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;
    log::trace!("[process_get_call] {}", url);
    let client_response = client.get(url).send().await?;

    if client_response.status() != StatusCode::OK {
        return Err(Box::from(format!(
            "[process_get_call] error status code {}",
            client_response.status()
        )));
    }
    log::trace!("[process_get_call] status {}", client_response.status());
    let response = client_response.bytes().await?;
    let result = str::from_utf8(&response)?;
    Ok(result.to_string())
}
