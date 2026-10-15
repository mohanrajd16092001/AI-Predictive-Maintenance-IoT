"""Project entry point. Run `python main.py --help` for all commands."""
import argparse
import contextlib
import json
import sys

from src.config import ROOT, prepare_directories


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description="Predictive maintenance using synthetic IoT data only.")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate reproducible fictional sensor histories")
    generate.add_argument("--machines", type=int, default=80)
    commands.add_parser("train", help="Train, evaluate and save the model and charts")
    simulation = commands.add_parser("simulate", help="Replay an unseen machine and simulate motor actions")
    simulation.add_argument("--machine", default=None)
    simulation.add_argument("--delay", type=float, default=0, help="Seconds between replay messages (default: fast)")
    predict = commands.add_parser("predict", help="Predict using three present sensor readings")
    for name in ["temperature", "vibration", "current"]:
        predict.add_argument(f"--{name}", type=float, required=True)
    commands.add_parser("run", help="Generate, train and simulate; save a complete run log")
    server = commands.add_parser("serve", help="Start the local dashboard and Flask API")
    server.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    prepare_directories()
    if args.command == "generate":
        from src.data import generate_data
        generate_data(args.machines)
    elif args.command == "train":
        from src.train import train_model
        train_model()
    elif args.command == "simulate":
        from src.simulate import simulate
        simulate(args.machine, args.delay)
    elif args.command == "predict":
        from src.predict import predict_reading
        print(json.dumps(predict_reading({name: getattr(args, name) for name in ["temperature", "vibration", "current"]}), indent=2))
    elif args.command == "run":
        from src.data import generate_data
        from src.train import train_model
        from src.simulate import simulate
        with (ROOT / "outputs" / "run.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                generate_data()
                train_model()
                simulate()
    elif args.command == "serve":
        from src.api import create_app
        create_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
