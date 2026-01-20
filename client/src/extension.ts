import * as path from 'path';
import { workspace, ExtensionContext, window, commands, StatusBarAlignment, StatusBarItem } from 'vscode';

import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  State,
} from 'vscode-languageclient/node';

let client: LanguageClient;
let outputTab: StatusBarItem;
let outputChannel = window.createOutputChannel('Dimspector');

export function activate(context: ExtensionContext) {
  // Create output channel tab (always visible, on LEFT side)
  outputTab = window.createStatusBarItem(
    StatusBarAlignment.Left,
    1 // priority (higher = more to the right of left tabs)
  );
  outputTab.command = 'dimspector.showOutput';
  outputTab.text = 'Dimspector';
  outputTab.tooltip = 'Show Dimspector output';
  outputTab.show();
  context.subscriptions.push(outputTab);

  // Register command to show output
  const showOutputCommand = commands.registerCommand('dimspector.showOutput', () => {
    outputChannel.show();
  });
  context.subscriptions.push(showOutputCommand);

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
    outputChannel: outputChannel,
    outputChannelName: 'Dimspector',
  };

  client = new LanguageClient(
    'dimspector',
    'Dimspector',
    serverOptions,
    clientOptions
  );

  // Register restart server command (after client is created)
  const restartCommand = commands.registerCommand('dimspector.restartServer', async () => {
    if (client) {
      await client.stop();
      await client.start();
    }
  });
  context.subscriptions.push(restartCommand);

  context.subscriptions.push(client);
  client.start();
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
