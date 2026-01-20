use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use tower_lsp::jsonrpc::Result as LSPResult;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use crate::api::Dimspector;

struct Backend {
    client: Client,
    project_root: Arc<RwLock<Option<PathBuf>>>,
    dimspector: Arc<RwLock<Option<Dimspector>>>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, params: InitializeParams) -> LSPResult<InitializeResult> {
        // Extract project root from workspace_folders
        if let Some(workspace_folders) = params.workspace_folders {
            if let Some(first_folder) = workspace_folders.first() {
                if let Ok(project_root) = first_folder.uri.to_file_path() {
                    self.client
                        .log_message(
                            MessageType::INFO,
                            format!("Workspace folder: {}", project_root.display()),
                        )
                        .await;
                    *self.project_root.write().unwrap() = Some(project_root);
                }
            }
        }

        let mut init_result = InitializeResult::default();
        init_result.capabilities.text_document_sync =
            Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL));
        init_result.capabilities.inlay_hint_provider = Some(OneOf::Left(true));
        Ok(init_result)
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "server initialized!")
            .await;

        // Run analysis: blocking work on thread pool, then async logging
        let client = self.client.clone();
        let dimspector = self.dimspector.clone();
        let project_root = self.project_root.clone();

        let (error_count, file_count) = tokio::task::spawn_blocking(move || {
            let project_root_guard = project_root.read().unwrap();
            let project_root = match project_root_guard.as_ref() {
                Some(root) => root,
                None => return (0, 0),
            };

            match Dimspector::from_project_root(project_root) {
                Ok((dimspector_instance, errors)) => {
                    // Count files from parsed project
                    let file_count = dimspector_instance.parsed_project.files.len();

                    // Log each error found
                    for (file_path, error) in &errors {
                        log::error!("Error in {}: {}", file_path.display(), error);
                    }
                    *dimspector.write().unwrap() = Some(dimspector_instance);
                    (errors.len(), file_count)
                }
                Err(e) => {
                    log::error!("Failed to initialize Dimspector: {:?}", e);
                    (0, 0)
                }
            }
        })
        .await
        .unwrap_or((0, 0));

        // Log results asynchronously
        client
            .log_message(
                MessageType::INFO,
                format!("Collected {} Python files from project", file_count),
            )
            .await;

        if error_count > 0 {
            client
                .log_message(
                    MessageType::INFO,
                    format!("Initial analysis completed with {} errors", error_count),
                )
                .await;
        } else {
            client
                .log_message(MessageType::INFO, "Initial analysis completed successfully")
                .await;
        }
    }

    async fn shutdown(&self) -> LSPResult<()> {
        Ok(())
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> LSPResult<Option<Vec<InlayHint>>> {
        // Extract file path from params
        let file_path = match params.text_document.uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                self.client
                    .log_message(MessageType::WARNING, "Failed to extract file path from URI")
                    .await;
                return Ok(None);
            }
        };

        self.client
            .log_message(
                MessageType::INFO,
                format!("Requesting inlay hints for: {}", file_path.display()),
            )
            .await;

        let file_path_display = file_path.display().to_string();

        // Get dimspector and call inlay_hints (drop lock before await)
        let hints = {
            let dimspector_guard = self.dimspector.read().unwrap();
            if let Some(dimspector) = dimspector_guard.as_ref() {
                let files = std::collections::HashSet::from([file_path]);
                Some(dimspector.inlay_hints(&files))
            } else {
                None
            }
        };

        match &hints {
            Some(hints_vec) => {
                self.client
                    .log_message(
                        MessageType::INFO,
                        format!(
                            "Returning {} inlay hints for: {}",
                            hints_vec.len(),
                            file_path_display
                        ),
                    )
                    .await;
                Ok(Some(hints_vec.clone()))
            }
            None => {
                self.client
                    .log_message(
                        MessageType::WARNING,
                        "Dimspector not initialized, cannot provide inlay hints",
                    )
                    .await;
                Ok(None)
            }
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let content = params.text_document.text;
        let version = params.text_document.version;

        // Analyze file and publish diagnostics
        self.analyze_and_publish_diagnostics(uri, content, Some(version))
            .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        let version = params.text_document.version;
        let client = self.client.clone();

        // Check if dimspector is initialized (drop guard before await)
        let is_initialized = {
            let dimspector_guard = self.dimspector.read().unwrap();
            dimspector_guard.is_some()
        };

        if !is_initialized {
            client
                .log_message(
                    MessageType::WARNING,
                    format!(
                        "Received did_change but Dimspector not initialized yet (file: {})",
                        uri
                    ),
                )
                .await;
            return;
        }

        // With TextDocumentSyncKind::FULL, content_changes[0].text is the full document
        if let Some(change) = params.content_changes.into_iter().next() {
            let content = change.text;
            let file_path_display = uri.to_string();

            client
                .log_message(
                    MessageType::INFO,
                    format!("did_change: {} (version: {})", file_path_display, version),
                )
                .await;

            // Re-analyze file and publish diagnostics
            self.analyze_and_publish_diagnostics(uri, content, Some(version))
                .await;
        } else {
            client
                .log_message(
                    MessageType::WARNING,
                    format!("did_change received with no content changes: {}", uri),
                )
                .await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        // Clear diagnostics for closed file
        self.client
            .publish_diagnostics(params.text_document.uri, vec![], None)
            .await;
    }
}

impl Backend {
    /// Analyze a file and publish diagnostics
    async fn analyze_and_publish_diagnostics(
        &self,
        uri: Url,
        content: String,
        version: Option<i32>,
    ) {
        // Extract file path
        let file_path = match uri.to_file_path() {
            Ok(path) => path,
            Err(_) => {
                self.client
                    .log_message(MessageType::WARNING, "Failed to extract file path from URI")
                    .await;
                return;
            }
        };

        let file_path_display = file_path.display().to_string();

        self.client
            .log_message(
                MessageType::INFO,
                format!("Analyzing file: {}", file_path_display),
            )
            .await;

        let client = self.client.clone();
        let dimspector = self.dimspector.clone();
        let uri_string = uri.as_str().to_string();

        // Run analysis in blocking task
        let (diagnostics, error_count) = tokio::task::spawn_blocking(move || {
            let mut dimspector_guard = dimspector.write().unwrap();

            // Check if dimspector is initialized
            let dimspector_instance = match dimspector_guard.as_mut() {
                Some(d) => d,
                None => {
                    log::warn!(
                        "analyze_and_publish_diagnostics: Dimspector not initialized for {}",
                        file_path.display()
                    );
                    return (Vec::new(), 0);
                }
            };

            log::info!("Starting analysis: {}", file_path.display());

            // Analyze the file
            match dimspector_instance.analyze_file(&file_path, &content) {
                Ok(errors) => {
                    let error_count = errors.len();
                    // Log each error found
                    for (error_file_path, error) in &errors {
                        log::error!("Error in {}: {}", error_file_path.display(), error);
                    }
                    // Convert ShapeErrors to LSP Diagnostics
                    let diagnostics: Vec<_> = errors
                        .into_iter()
                        .filter_map(|(_, error)| error.to_diagnostic(&content, &uri_string))
                        .collect();
                    log::info!(
                        "Analysis complete: {} errors found in {}",
                        error_count,
                        file_path.display()
                    );
                    (diagnostics, error_count)
                }
                Err(e) => {
                    log::error!("Failed to analyze file {}: {:?}", file_path.display(), e);
                    (Vec::new(), 0)
                }
            }
        })
        .await
        .unwrap_or_else(|e| {
            log::error!("Analysis task panicked: {:?}", e);
            (Vec::new(), 0)
        });

        // Log results
        if error_count > 0 {
            client
                .log_message(
                    MessageType::INFO,
                    format!(
                        "Found {} errors in {} (published {} diagnostics)",
                        error_count,
                        file_path_display,
                        diagnostics.len()
                    ),
                )
                .await;
        } else {
            client
                .log_message(
                    MessageType::INFO,
                    format!("No errors found in {}", file_path_display),
                )
                .await;
        }

        // Publish diagnostics
        client.publish_diagnostics(uri, diagnostics, version).await;
    }
}

#[tokio::main]
pub async fn start_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        project_root: Arc::new(RwLock::new(None)),
        dimspector: Arc::new(RwLock::new(None)),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
