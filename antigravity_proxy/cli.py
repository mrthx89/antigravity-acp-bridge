"""Command-line entry point for Antigravity Proxy."""

from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser

from .auth import ProjectResolver, TokenManager
from .config import MODEL_MAP, ProxyConfig
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
        description="Antigravity OpenAI-compatible proxy (Cloud Code Assist API)"
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
        api_key=args.api_key,
        allow_unauthenticated=args.allow_unauthenticated,
        cors_origins=args.cors_origin,
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
    server = ProxyHTTPServer(
        (config.host, config.port),
        config,
        token_manager,
        project_resolver,
        upstream_client,
    )

    logger.info("Antigravity proxy listening on http://%s:%s/v1", config.host, config.port)
    logger.info("  Auth:   %s", "Bearer/x-api-key enabled" if config.api_key else "disabled (localhost mode)")
    logger.info("  CORS:   %s", config.cors_origins)
    logger.info("  Models: %s", ", ".join(MODEL_MAP.keys()))
    logger.info("  Token:  %s", config.token_file)
    logger.info("  Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()
    finally:
        server.server_close()

    return 0
