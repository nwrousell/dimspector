use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::Result;
use tower_lsp::jsonrpc::{Error as LSPError, Result as LSPResult};
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use crate::analysis::{self};
use crate::{ast, ir};

#[derive(Debug)]
struct FileState {
    version: i32,
    content: String,
    is_open: bool,
}

#[derive(Debug)]
struct Backend {
    client: Client,
    files: RwLock<HashMap<Url, FileState>>,
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> LSPResult<InitializeResult> {
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
    }

    async fn shutdown(&self) -> LSPResult<()> {
        Ok(())
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> LSPResult<Option<Vec<InlayHint>>> {
        // Get content of file, return None if not tracked yet
        let contents = {
            let files = self.files.read().unwrap();
            match files.get(&params.text_document.uri) {
                Some(state) => state.content.clone(),
                None => return Ok(None),
            }
        };

        let input = ast::Input {
            contents,
            path: params.text_document.uri.path().into(),
        };

        // Run CPU-bound analysis on blocking thread pool
        let result = tokio::task::spawn_blocking(move || single_file_to_inlay_hints(input))
            .await
            .map_err(|e| LSPError {
                code: tower_lsp::jsonrpc::ErrorCode::InternalError,
                message: format!("task panicked: {e}").into(),
                data: None,
            })?;

        match result {
            Ok(hints) => Ok(Some(hints)),
            Err(err) => Err(LSPError {
                code: tower_lsp::jsonrpc::ErrorCode::InternalError,
                message: err.to_string().into(),
                data: None,
            }),
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;
        let content = params.text_document.text;

        {
            let mut files = self.files.write().unwrap();
            files.insert(
                uri.clone(),
                FileState {
                    version,
                    content: content.clone(),
                    is_open: true,
                },
            );
        }

        // self.client
        //     .log_message(MessageType::INFO, format!("did_open: {} v{}", uri, version))
        //     .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let version = params.text_document.version;

        // With TextDocumentSyncKind::FULL, content_changes[0].text is the full document
        if let Some(change) = params.content_changes.into_iter().next() {
            let mut files = self.files.write().unwrap();
            if let Some(state) = files.get_mut(&uri) {
                state.version = version;
                state.content = change.text;
            }
        }

        // self.client
        //     .log_message(
        //         MessageType::INFO,
        //         format!("did_change: {} v{}", uri, version),
        //     )
        //     .await;
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;

        {
            let mut files = self.files.write().unwrap();
            if let Some(state) = files.get_mut(&uri) {
                state.is_open = false;
            }
        }

        // self.client
        //     .log_message(MessageType::INFO, format!("did_close: {}", uri))
        //     .await;
    }
}

#[tokio::main]
pub async fn start_server() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        files: RwLock::new(HashMap::new()),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}

fn single_file_to_inlay_hints(input: ast::Input) -> Result<Vec<InlayHint>> {
    let line_index = ast::LineIndex::new(&input.contents);
    let prog = ast::parse(&input)?;
    let ir = ir::lower(prog, &line_index)?;
    let analysis = analysis::analyze(ir)?;
    Ok(analysis.inlay_hints())
}
