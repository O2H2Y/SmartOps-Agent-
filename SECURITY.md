# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security vulnerabilities. Use GitHub's
private vulnerability reporting for this repository and include:

- the affected component and version;
- reproduction steps;
- potential impact;
- any suggested mitigation.

## Local development warning

SmartOps is currently a local-first reference implementation. The HTTP API does
not include authentication, and the included Compose file uses development-only
credentials by default. Do not expose the application, Milvus, MinIO, or MCP
services directly to an untrusted network without adding authentication,
network restrictions, TLS, and non-default credentials.

Never commit a real `.env` file, API key, cloud credential, private key, access
log, uploaded document, or local database snapshot.
