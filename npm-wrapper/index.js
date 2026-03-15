#!/usr/bin/env node

/**
 * emms-mcp — npm wrapper for the EMMS MCP server (Python).
 *
 * Tries these methods in order:
 *   1. uvx emms-mcp    (fastest, no prior install needed)
 *   2. pipx run emms-mcp
 *   3. python -m emms.mcp_entry
 *
 * All CLI arguments are forwarded to the Python process.
 * stdio is inherited so MCP transport works transparently.
 */

const { spawn, execSync } = require("child_process");

const args = process.argv.slice(2);

function commandExists(cmd) {
  try {
    execSync(`which ${cmd}`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function run(command, commandArgs) {
  const child = spawn(command, commandArgs, {
    stdio: "inherit",
    env: process.env,
  });

  child.on("error", (err) => {
    process.stderr.write(`Failed to start ${command}: ${err.message}\n`);
    process.exit(1);
  });

  child.on("exit", (code) => {
    process.exit(code ?? 1);
  });
}

if (commandExists("uvx")) {
  run("uvx", ["emms-mcp", ...args]);
} else if (commandExists("pipx")) {
  run("pipx", ["run", "emms-mcp", ...args]);
} else if (commandExists("python3")) {
  run("python3", ["-m", "emms.mcp_entry", ...args]);
} else if (commandExists("python")) {
  run("python", ["-m", "emms.mcp_entry", ...args]);
} else {
  process.stderr.write(
    "Error: emms-mcp requires Python. Install uvx, pipx, or python3.\n" +
      "  pip install emms-mcp   # then run: emms-mcp\n" +
      "  pip install uv         # then run: npx emms-mcp\n"
  );
  process.exit(1);
}
