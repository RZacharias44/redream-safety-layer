# Langfuse Tracing

The chat application still has Langfuse tracing instrumentation in `AI/irt_app.py`.
Tracing is configured from environment variables after `.env` is loaded:

```env
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
LANGFUSE_ENABLED=true
```

`LANGFUSE_ENABLED=false` disables tracing without removing the decorators.

## What Is Traced

- `process_chat_message` and `process_chat_message_stream` create top-level chat traces.
- `determine_stage_async` records stage-routing observations.
- `generate_nightmare_summary` records nightmare-summary generation.
- `evaluate_safety` records safety-critic evaluation.
- `get_response_async` and `get_response_stream_async` record final response generation.

Most decorators use `capture_input=False` and `capture_output=False`, then update
Langfuse manually with selected inputs, outputs, metadata, session IDs, user IDs,
stage tags, and token usage.

## Self-Hosting On A Scaleway VM

For a low-scale thesis/company instance, use the official Langfuse Docker Compose
deployment on an Ubuntu VM. Size the VM at least around the official baseline:
4 vCPU, 16 GiB RAM, and enough persistent disk for trace growth, for example
100 GiB or more.

Recommended setup:

1. Create an Ubuntu VM in the company Scaleway project.
2. Attach persistent storage and make sure Docker volumes live on it.
3. Install Docker and the Docker Compose plugin.
4. Clone `https://github.com/langfuse/langfuse.git`.
5. Generate new secrets for every `# CHANGEME` value in `docker-compose.yml`.
6. Set `NEXTAUTH_URL` to the final HTTPS URL, for example
   `https://langfuse.example.com`.
7. Put a reverse proxy or Scaleway Load Balancer in front of port `3000` and
   terminate TLS there.
8. Restrict inbound network access. The web UI/API needs to be reachable by this
   app; Postgres, ClickHouse, Redis, and internal object storage should not be
   public.
9. Start Langfuse with `docker compose up -d`.
10. Create an organization/project in the Langfuse UI and copy the project keys.
11. Update this app's `.env` with the new host and keys, then set
    `LANGFUSE_ENABLED=true`.

## Validation

Start the app:

```bash
uv run python -m AI.api
```

Send a message to `/chat` or `/chat/stream`, then check the Langfuse project for
a trace named `Chat Session: ...` or `Streaming Chat Session: ...`.

If no trace appears:

- Confirm `LANGFUSE_ENABLED=true`.
- Confirm `LANGFUSE_HOST` is the external URL without a trailing path.
- Confirm the project public and secret keys are from the new self-hosted
  instance.
- Check app logs for Langfuse connection errors.
