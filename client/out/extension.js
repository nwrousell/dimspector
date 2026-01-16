"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const path = require("path");
const vscode_1 = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
function activate(context) {
    // Check for user-configured path first
    const configuredPath = vscode_1.workspace.getConfiguration('dimspector').get('serverPath');
    let serverCommand;
    if (configuredPath) {
        serverCommand = configuredPath;
    }
    else {
        // Default: look for cargo build output relative to extension directory
        // Extension is in `client/`, binary is in `target/debug/`
        const binaryName = process.platform === 'win32' ? 'dimspector.exe' : 'dimspector';
        serverCommand = path.join(context.extensionPath, '..', 'target', 'debug', binaryName);
    }
    const serverOptions = {
        command: serverCommand,
        args: ['server'],
        // Uses stdio by default when you specify `command`
    };
    const clientOptions = {
        // Register the server for Python files
        documentSelector: [{ scheme: 'file', language: 'python' }],
    };
    client = new node_1.LanguageClient('dimspector', 'Dimspector', serverOptions, clientOptions);
    client.start();
}
function deactivate() {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
//# sourceMappingURL=extension.js.map