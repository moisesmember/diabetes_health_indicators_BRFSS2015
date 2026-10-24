"""Interface de linha de comando: ``diabetes-risk <etapa> [--config params.yaml]``.

Etapas:
    prepare   valida e divide os dados brutos;
    train     treina o modelo e escolhe o limiar;
    evaluate  avalia no teste e aplica o quality gate (exit code 1 se reprovar);
    run       executa prepare, train e evaluate em sequência;
    serve     sobe a API de inferência.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from diabetes_risk.config import load_config

EXIT_GATE_FAILED = 1


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos."""
    parser = argparse.ArgumentParser(prog="diabetes-risk", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="params.yaml", help="arquivo YAML de configuração")
    parser.add_argument("-v", "--verbose", action="store_true", help="logs em nível DEBUG")

    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("prepare", help="valida e divide os dados brutos")
    stages.add_parser("train", help="treina o modelo")
    stages.add_parser("evaluate", help="avalia o modelo e aplica o quality gate")
    stages.add_parser("run", help="executa prepare, train e evaluate em sequência")
    serve = stages.add_parser("serve", help="sobe a API de inferência")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada da CLI. Retorna o exit code do processo."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config = load_config(args.config)

    # Imports tardios: cada etapa carrega só o que usa (ex.: `serve` não importa o treino).
    if args.stage in ("prepare", "run"):
        from diabetes_risk.pipelines.prepare import run_prepare

        run_prepare(config)
    if args.stage in ("train", "run"):
        from diabetes_risk.pipelines.train import run_train

        run_train(config)
    if args.stage in ("evaluate", "run"):
        from diabetes_risk.pipelines.evaluate import run_evaluate

        if not run_evaluate(config).gate.passed:
            return EXIT_GATE_FAILED
    if args.stage == "serve":
        import uvicorn

        from diabetes_risk.models.bundle import ModelBundle
        from diabetes_risk.serving.api import create_app

        app = create_app(ModelBundle.load(config.artifacts.model_path))
        uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
