import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';

import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
  // Check for user-configured path first
  const configuredPath = workspace.getConfiguration('dimspector').get<string>('serverPath');
  
  let serverCommand: string;
  if (configuredPath) {
    serverCommand = configuredPath;
  } else {
    // Default: look for cargo build output relative to extension directory
    // Extension is in `client/`, binary is in `target/debug/`
    const binaryName = process.platform === 'win32' ? 'dimspector.exe' : 'dimspector';
    serverCommand = path.join(context.extensionPath, '..', 'target', 'debug', binaryName);
  }

  const serverOptions: ServerOptions = {
    command: serverCommand,
    args: ['server'],
    // Uses stdio by default when you specify `command`
  };

  const clientOptions: LanguageClientOptions = {
    // Register the server for Python files
    documentSelector: [{ scheme: 'file', language: 'python' }],
  };

  client = new LanguageClient(
    'dimspector',
    'Dimspector',
    serverOptions,
    clientOptions
  );

  client.start();
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
