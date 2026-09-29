# Security policy

## Supported versions

Only the latest commit on `main` is supported during the alpha stage.

## Reporting

Do not open a public issue for leaked credentials or a vulnerability that could
expose user data. Use GitHub's private vulnerability reporting after the repository
is published, or contact the repository owner privately.

## Data and model safety

- The browser adapter blocks network requests and disables page JavaScript.
- The OpenAI-compatible VLM adapter sends supplied images and prompts to the configured endpoint.
- Keep API keys in environment variables; never add `.env` files to Git.
- Treat downloaded datasets, model weights, HTML, and VLM output as untrusted input.
- Do not run unknown generated HTML with network access or JavaScript enabled.
