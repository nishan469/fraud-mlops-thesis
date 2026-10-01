"""Command line: python -m fraud_mlops <command> [--config configs/default.toml]

  replay    run the continuous-learning loop over the historical stream
  champion  show the currently promoted model
  dashboard build the monitoring dashboard (HTML) for the last replay
  serve     start the scoring API (FastAPI + uvicorn)
"""

import argparse
import json

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraud_mlops")
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--data_dir", help="override [data] data_dir")
    ap.add_argument("--out_dir", help="override [replay] out_dir")
    sub = ap.add_subparsers(dest="command", required=True)

    rp = sub.add_parser("replay", help="replay the stream through monitor/retrain/gate/serve")
    rp.add_argument("--label_delay", type=float, help="override [policy] label_delay_days")
    rp.add_argument("--end_day", type=float, help="stop the replay at this day")

    sub.add_parser("champion", help="show the current champion model")

    dp = sub.add_parser("dashboard", help="build the monitoring dashboard for the last replay")
    dp.add_argument("--out", help="output HTML path (default: <replay out_dir>/dashboard.html)")

    sp = sub.add_parser("serve", help="start the scoring API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)
    overrides = {}
    if getattr(args, "label_delay", None) is not None:
        overrides["policy"] = {"label_delay_days": args.label_delay}
    if args.data_dir:
        overrides["data"] = {"data_dir": args.data_dir}
    if args.out_dir:
        overrides["replay"] = {"out_dir": args.out_dir}
    cfg = load_config(args.config, overrides)

    if args.command == "replay":
        from .pipeline import ReplayRunner
        summary, _, _ = ReplayRunner(cfg).run(end_day=args.end_day)
        print("\n=== Replay summary ===\n" + json.dumps(summary, indent=2))
        print(f"Outputs in {cfg.replay.out_dir}; browse runs with: "
              f"mlflow ui --backend-store-uri {cfg.mlflow.tracking_uri}")

    elif args.command == "champion":
        from .registry import ModelRegistry
        reg = ModelRegistry(cfg.mlflow)
        version = reg.champion_version()
        if version is None:
            print("No champion yet.")
            return
        mv = reg.client.get_model_version(reg.name, version)
        print(json.dumps({"model": reg.name, "version": version, "tags": mv.tags}, indent=2))

    elif args.command == "dashboard":
        from .dashboard import build_dashboard
        print(f"Dashboard written to {build_dashboard(cfg, args.out)}")

    elif args.command == "serve":
        import uvicorn
        from .serving import create_app
        uvicorn.run(create_app(cfg), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
