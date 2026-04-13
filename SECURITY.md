# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email: **security@costguard.dev**

We will respond within 48 hours and aim to release a fix within 7 days of confirmation.

## Security Design

- **No data persistence**: Uploaded files are processed in memory and never written to permanent storage.
- **No user accounts**: CostGuard collects no personally identifiable information.
- **API keys**: Never logged, never stored beyond the runtime process.
- **File validation**: Uploaded files are type-checked and size-limited before processing.
- **Non-root Docker**: The container runs as a non-root user (UID 1001).
- **CORS**: Strict origin allowlist configured via `CORS_ORIGINS` env var.
