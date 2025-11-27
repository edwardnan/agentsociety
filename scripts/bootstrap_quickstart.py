"""Bootstrap assets for an out-of-the-box AgentSociety run.

The script prepares three things that normally block new users:

- Downloads the Beijing map file to ``data/beijing_map.pb``.
- Generates simulation and experiment config files with sensible defaults
  that point to local Docker services.
- Emits an optional UI config so the web UI can connect to the same services.

Only the LLM API key is required. Everything else has defaults matching the
Docker compose bundle in ``docker/``.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from urllib import request


DEFAULT_MAP_URL = (
    "https://cloud.tsinghua.edu.cn/f/f5c777485d2748fa8535/?dl=1"
)


def _prompt_secret(prompt: str) -> str:
    try:
        return getpass.getpass(prompt)
    except (EOFError, KeyboardInterrupt):
        return ""


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _download_map(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✔ Map already exists at {dest} (skip download)")
        return

    print(f"⬇️  Downloading city map from {url} ...")
    try:
        request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001 - want to show the raw error
        raise SystemExit(f"Failed to download map: {exc}") from exc
    else:
        print(f"✔ Saved map to {dest}")


def _build_sim_config(args: argparse.Namespace, map_path: Path, api_key: str) -> str:
    total_step = args.max_day * 86400
    mqtt_block = dedent(
        f"""
        mqtt:
          server: {args.mqtt_host}
          port: {args.mqtt_port}
          username: {args.mqtt_username}
          password: {args.mqtt_password}
        """
    )
    if args.offline:
        mqtt_block = dedent(
            f"""
            mqtt:
              mode: local
              local_path: {args.offline_log_path}
            """
        )

    metric_block = dedent(
        f"""
        metric_request:
          mlflow:
              username: {args.mlflow_username}
              password: {args.mlflow_password}
              mlflow_uri: {args.mlflow_uri}
        """
    )
    if args.offline:
        metric_block = "metric_request: {}\n"

    pgsql_block = dedent(
        f"""
        pgsql:
          enabled: {str(not args.offline).lower()}
          dsn: {args.pgsql_dsn if not args.offline else 'postgresql://offline'}
        """
    )

    return dedent(
        f"""
        llm_request:
          request_type: {args.llm_provider}
          api_key: {api_key}
          model: {args.llm_model}

        simulator_request:
          task_name: "{args.task_name}"
          max_day: {args.max_day}
          start_step: 0
          total_step: {total_step}
          log_dir: log
          min_step_time: 1000

        {mqtt_block}

        map_request:
          file_path: {map_path}

        {metric_block}

        {pgsql_block}

        avro:
          enabled: true
          path: {args.avro_path}
        """
    ).strip() + "\n"


def _build_exp_config(args: argparse.Namespace) -> str:
    return dedent(
        f"""
        agent_config:
          number_of_citizen: {args.number_of_citizen}

        workflow: [
          {{"type": "run", "days": {args.max_day}}}
        ]
        """
    ).strip() + "\n"


def _build_ui_config(args: argparse.Namespace, map_path: Path) -> str:
    return dedent(
        f"""
        mqtt:
          host: {args.mqtt_host}
          port: {args.mqtt_port}
          username: {args.mqtt_username}
          password: {args.mqtt_password}

        database:
          dsn: {args.pgsql_dsn}

        mlflow:
          username: {args.mlflow_username}
          password: {args.mlflow_password}
          tracking_uri: {args.mlflow_uri}

        map_request:
          file_path: {map_path}
        """
    ).strip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare required assets (map + configs) to run the AgentSociety "
            "quickstart example with minimal manual editing."
        )
    )

    parser.add_argument("--llm-provider", default="zhipuai")
    parser.add_argument("--llm-model", default="GLM-4-Flash")
    parser.add_argument("--llm-api-key", help="API key for your LLM provider")
    parser.add_argument("--map-url", default=DEFAULT_MAP_URL)
    parser.add_argument(
        "--output-dir",
        default="quickstart",
        help="Directory to place generated configs",
    )
    parser.add_argument(
        "--map-path",
        default="data/beijing_map.pb",
        help="Where to save the downloaded map",
    )
    parser.add_argument("--task-name", default="citysim")
    parser.add_argument("--max-day", type=int, default=1)
    parser.add_argument("--number-of-citizen", type=int, default=100)

    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-username", default="admin")
    parser.add_argument("--mqtt-password", default="public")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Generate configs that avoid Docker services by using a local JSONL messager and disabling MLflow/PostgreSQL",
    )
    parser.add_argument(
        "--offline-log-path",
        default="cache/offline_messages.jsonl",
        help="Where to write local message traces when --offline is set",
    )

    parser.add_argument("--mlflow-uri", default="http://localhost:59000")
    parser.add_argument("--mlflow-username", default="admin")
    parser.add_argument("--mlflow-password", default="change_me")

    parser.add_argument(
        "--pgsql-dsn",
        default="postgresql://postgres:CHANGE_ME@localhost:5432/postgres",
    )

    parser.add_argument("--avro-path", default="cache/avro")
    parser.add_argument(
        "--skip-map-download",
        action="store_true",
        help="Assume the map already exists at --map-path",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="After generating configs, automatically launch docker services and run the quickstart",
    )
    parser.add_argument(
        "--no-run-prompt",
        action="store_true",
        help="Do not ask to run the quickstart interactively (implied when --run is set)",
    )
    parser.add_argument(
        "--skip-docker-start",
        action="store_true",
        help="Do not call `docker compose up -d` before running the quickstart",
    )

    return parser.parse_args(argv)


def _start_docker_services() -> bool:
    if shutil.which("docker") is None:
        print(
            "✖ Docker is not installed or not in PATH. Install Docker Desktop/Engine "
            "and rerun, or pass --skip-docker-start to manage services yourself."
        )
        return False

    compose_cmd = ["docker", "compose"]
    result = subprocess.run(compose_cmd + ["version"], capture_output=True)
    if result.returncode != 0:
        print(
            "✖ `docker compose` is unavailable. Ensure Docker Compose V2 is installed, "
            "or start the services manually inside the docker/ folder."
        )
        return False

    print("▶ Starting docker services (this may take a bit on first run)...")
    try:
        subprocess.run(compose_cmd + ["up", "-d"], cwd=Path("docker"), check=True)
    except subprocess.CalledProcessError as exc:  # noqa: BLE001 - surface docker error
        print(f"✖ Failed to start docker services: {exc}")
        return False
    else:
        print("✔ Docker services are up")
        return True


async def _run_quickstart(sim_config_path: Path, exp_config_path: Path) -> None:
    from agentsociety import AgentSimulation
    from agentsociety.configs import ExpConfig, SimConfig, load_config_from_file

    sim_config = load_config_from_file(str(sim_config_path), SimConfig)
    exp_config = load_config_from_file(str(exp_config_path), ExpConfig)

    await AgentSimulation.run_from_config(exp_config, sim_config)


def _maybe_run_quickstart(
    sim_config_path: Path,
    exp_config_path: Path,
    *,
    force_run: bool,
    prompt_user: bool,
    skip_docker: bool,
) -> None:
    if not force_run:
        if not prompt_user:
            print(
                "Next step: start docker services (see docker/README.md), then run your "
                "simulation pointing to these configs."
            )
            return
        if not _prompt_yes_no(
            "Start docker services and launch a one-day quickstart run now?", default=True
        ):
            print(
                "Skipping auto-run. Start services with `docker compose up -d` and run "
                "your simulation when ready."
            )
            return

    if not skip_docker:
        started = _start_docker_services()
        if not started:
            return
    else:
        print("Skipping docker startup as requested. Make sure services are already running.")

    print("▶ Launching quickstart simulation... (Ctrl+C to stop)")
    try:
        asyncio.run(_run_quickstart(sim_config_path, exp_config_path))
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = args.llm_api_key or _prompt_secret("Enter your LLM API key: ")
    if not api_key:
        print("✖ LLM API key is required. Provide --llm-api-key or type it interactively.")
        return 1

    map_path = Path(args.map_path)
    if not args.skip_map_download:
        _download_map(args.map_url, map_path)
    elif not map_path.exists():
        print(f"✖ --skip-map-download is set but {map_path} does not exist.")
        return 1

    output_dir = Path(args.output_dir)
    sim_config_path = output_dir / "sim_config.yaml"
    exp_config_path = output_dir / "exp_config.yaml"
    ui_config_path = output_dir / "ui_config.yaml"

    _write_text(sim_config_path, _build_sim_config(args, map_path, api_key))
    _write_text(exp_config_path, _build_exp_config(args))
    _write_text(ui_config_path, _build_ui_config(args, map_path))

    print("✔ Generated configs:")
    print(f"  - {sim_config_path}")
    print(f"  - {exp_config_path}")
    print(f"  - {ui_config_path}")
    if args.offline:
        print(
            "Offline mode: MQTT traffic will be logged locally and MLflow/PostgreSQL are disabled."
        )
        args.skip_docker_start = True
    _maybe_run_quickstart(
        sim_config_path,
        exp_config_path,
        force_run=args.run,
        prompt_user=not args.no_run_prompt,
        skip_docker=args.skip_docker_start,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
