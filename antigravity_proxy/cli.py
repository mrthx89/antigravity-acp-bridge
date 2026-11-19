"""Command-line entry point for Antigravity Proxy."""

from __future__ import annotations

import logging
import signal
import sys
import threading
from argparse import ArgumentParser

from .agy_cli import AgyCliClient
from .auth import ProjectResolver, TokenManager
from .config import MODEL_MAP, ProxyConfig, SERVICE_TITLE, SERVICE_VERSION, TRADEMARK
from .server import ProxyHTTPServer
from .upstream import UpstreamClient

logger = logging.getLogger("antigravity_proxy")


def configure_logging(level_name: str = "INFO") -> None:
    """Configure clean stderr logging for the proxy process."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[antigravity-proxy] %(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def build_arg_parser() -> ArgumentParser:
    """Create the CLI argument parser."""
    parser = ArgumentParser(
        description=f"{SERVICE_TITLE} v{SERVICE_VERSION} — {TRADEMARK}"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: 8877 or ANTIGRAVITY_PROXY_PORT)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: 127.0.0.1 or ANTIGRAVITY_PROXY_HOST)",
    )
    parser.add_argument(
        "--backend",
        choices=["agy-cli", "oauth"],
        default=None,
        help="Execution backend: agy-cli (default, official local agy CLI) or oauth (legacy direct API)",
    )
    parser.add_argument(
        "--agy-bin",
        default=None,
        help="Path or command name for agy executable when using --backend agy-cli (default: agy)",
    )
    parser.add_argument(
        "--agy-cwd",
        default=None,
        help="Working directory for agy-cli calls (default: isolated empty temp directory)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max concurrent agy-cli subprocesses, 0 for unlimited (default: 2)",
    )
    parser.add_argument(
        "--worker-idle-timeout",
        type=float,
        default=None,
        help=(
            "Seconds an idle agy-cli worker stays warm before it is closed "
            "(or ANTIGRAVITY_WORKER_IDLE_TIMEOUT, default: 300; 0 disables)"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require Bearer / x-api-key auth on API routes (or ANTIGRAVITY_PROXY_API_KEY)",
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="Allow binding to a non-localhost address without --api-key (unsafe)",
    )
    parser.add_argument(
        "--cors-origin",
        default=None,
        help="Access-Control-Allow-Origin value or comma-separated list (default: *)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI options, verify OAuth credentials, and run the HTTP proxy."""
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)

    config = ProxyConfig.from_env(
        host=args.host,
        port=args.port,
        backend=args.backend,
        api_key=args.api_key,
        allow_unauthenticated=args.allow_unauthenticated,
        cors_origins=args.cors_origin,
        agy_bin=args.agy_bin,
        agy_cwd=args.agy_cwd,
        max_concurrency=args.max_concurrency,
        worker_idle_timeout=args.worker_idle_timeout,
    )

    if not config.is_localhost and not config.api_key:
        if not config.allow_unauthenticated:
            logger.error(
                "Refusing to bind non-localhost host '%s' without --api-key "
                "(or ANTIGRAVITY_PROXY_API_KEY). Pass --allow-unauthenticated to override.",
                config.host,
            )
            return 2
        logger.warning(
            "Binding to '%s' WITHOUT API key authentication — anyone reachable can use your quota.",
            config.host,
        )

    token_manager = None
    project_resolver = None
    upstream_client = None
    agy_client = None

    if config.backend == "agy-cli":
        agy_client = AgyCliClient(config)
        if not agy_client.preflight_check():
            return 1
        try:
            agy_client.warm_up()
        except Exception as exc:
            logger.warning("Could not pre-warm ACP worker: %s", exc)
        models_summary = ", ".join(agy_client.list_models())
    else:
        token_manager = TokenManager(config)
        if not token_manager.preflight_check():
            return 1

        project_resolver = ProjectResolver(config, token_manager)
        try:
            project_resolver.get_project_id()
        except Exception as exc:
            logger.warning("Could not pre-discover project ID: %s", exc)
            logger.info("Project ID will be discovered on the first request.")

        upstream_client = UpstreamClient(token_manager, timeout=config.upstream_timeout)
        models_summary = ", ".join(MODEL_MAP.keys())

    server = ProxyHTTPServer(
        (config.host, config.port),
        config,
        token_manager=token_manager,
        project_resolver=project_resolver,
        upstream_client=upstream_client,
        agy_client=agy_client,
    )

    logger.info("======================================================================")
    logger.info("%s v%s — %s", SERVICE_TITLE, SERVICE_VERSION, TRADEMARK)
    logger.info("======================================================================")
    logger.info("Listening on http://%s:%s/v1", config.host, config.port)
    logger.info("  Platform: %s", sys.platform)
    logger.info("  Backend:  %s", config.backend)
    logger.info("  Auth:     %s", "Bearer/x-api-key enabled" if config.api_key else "disabled (localhost mode)")
    logger.info("  CORS:     %s", config.cors_origins)
    logger.info("  Models:   %s", models_summary)
    if config.backend == "agy-cli":
        workers_label = "unlimited" if config.max_concurrency <= 0 else config.max_concurrency
        logger.info("  Workers:  max_concurrency=%s", workers_label)
        idle_label = (
            f"reap after {config.worker_idle_timeout:.0f}s"
            if config.worker_idle_timeout > 0
            else "never reaped"
        )
        logger.info("  Idle:     %s", idle_label)
    else:
        logger.info("  Token:    %s", config.token_file)
    logger.info("  Press Ctrl+C to stop")

    prev_sigterm = None
    has_sigterm = hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread()
    if has_sigterm:
        def _handle_sigterm(signum, frame):
            logger.info("Received SIGTERM, shutting down...")
            threading.Thread(target=server.shutdown, daemon=True).start()

        prev_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()
    finally:
        if has_sigterm and prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)
        if agy_client is not None:
            agy_client.close()
        server.server_close()

    return 0
