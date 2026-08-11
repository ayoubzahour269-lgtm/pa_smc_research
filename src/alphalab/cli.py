"""Point d'entree unique de la plateforme.

    alphalab status                 etat des donnees disponibles
    alphalab spread XAUUSD --tf H1  profil de cout reel (mediane, pic horaire)
    alphalab freeze EURUSD ...      telechargement + gel (necessite un acces reseau)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from alphalab import __version__
from alphalab.config import (
    HOLDOUT_START,
    N_BOOTSTRAP,
    N_RANDOM_CONTROL,
    UNIVERSE,
    in_sample_start,
)
from alphalab.data import costs, registry


def _print_table(rows: list[dict[str, object]], empty: str) -> None:
    if not rows:
        print(empty)
        return
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_status(_: argparse.Namespace) -> int:
    print(f"alphalab {__version__}")
    print(f"Hold-out : >= {HOLDOUT_START.date()} — ouvert UNE seule fois, a la fin du projet.\n")

    print("=== Donnees disponibles ===")
    _print_table(registry.summary(), "Aucun snapshot. Voir : alphalab freeze --help")

    symbols = registry.available_symbols()
    if symbols:
        print("\n=== Fenetres in-sample ===")
        _print_table(
            [
                {
                    "symbole": s,
                    "debut": str(in_sample_start(s).date()),
                    "fin": str(HOLDOUT_START.date()),
                }
                for s in symbols
            ],
            "",
        )

    manquants = [s for s in UNIVERSE if s not in symbols]
    if manquants:
        print("\n=== Symboles connus mais absents du disque ===")
        for s in manquants:
            print(
                f"  {s:8s} {UNIVERSE[s].label:20s} -> "
                f"alphalab freeze {s} --tf M15 --side BID,ASK"
            )
        print(
            "\nNote : le facteur dollar (correlation or <-> EUR/USD) exige EURUSD, "
            "et idealement GBPUSD + USDJPY. Sans eux, il est declare indisponible."
        )
    return 0


def cmd_spread(args: argparse.Namespace) -> int:
    try:
        model = costs.cost_model(args.symbol, args.tf)
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1
    print(f"=== Cout reel {args.symbol} {args.tf} ===")
    _print_table([model.describe()], "")
    print("\n=== Profil par heure UTC ===")
    print(costs.hourly_profile(model.spread).to_string())
    print(
        "\nLecture : la colonne x_mediane_globale reperant les creneaux structurellement "
        "chers. Une strategie qui n'exploite que ces creneaux est un artefact de cout."
    )
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    from alphalab.data.freeze import FetchError, freeze

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    code = 0
    for tf in args.tf.split(","):
        for side in args.side.split(","):
            label = f"{args.symbol} {tf} {side}"
            try:
                out = freeze(
                    args.symbol, tf.strip(), side.strip(), start, end, version=args.version
                )
            except FetchError as exc:
                print(f"[ECHEC] {label} : {exc}", file=sys.stderr)
                code = 1
                continue
            if out.get("skipped"):
                print(f"[IGNORE] {out['basename']} — {out['reason']}")
            else:
                print(f"[GELE]   {out['basename']} — sha256 {str(out['sha256'])[:12]}")
    return code


def cmd_explore(args: argparse.Namespace) -> int:
    from alphalab import explore
    from alphalab.report import research

    symbols = args.symbols or registry.tradable(args.tf)
    if not symbols:
        print(
            f"Aucun symbole tradable en {args.tf} (il faut les cotes BID ET ASK).",
            file=sys.stderr,
        )
        return 1

    print(f"Exploration de {len(symbols)} symbole(s) en {args.tf}...", flush=True)
    result = explore.run(
        symbols,
        args.tf,
        n_control=args.controls,
        n_boot=args.bootstrap,
        alpha=args.alpha,
        progress=args.verbose,
    )
    print()
    print(research.to_console(result))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(research.to_markdown(result), encoding="utf-8")
        # Table brute a cote du rapport : permet de re-rendre ou de reanalyser sans
        # relancer une campagne de plusieurs minutes.
        csv_path = path.with_suffix(".csv")
        result.table.to_csv(csv_path, index=False)
        print(f"\nRapport ecrit : {path}\nTable brute   : {csv_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphalab", description=__doc__)
    parser.add_argument("--version", action="version", version=f"alphalab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="etat des donnees et des fenetres")
    p_status.set_defaults(func=cmd_status)

    p_spread = sub.add_parser("spread", help="profil de cout reel d'un symbole")
    p_spread.add_argument("symbol")
    p_spread.add_argument("--tf", default="H1")
    p_spread.set_defaults(func=cmd_spread)

    p_freeze = sub.add_parser("freeze", help="telecharge et gele un snapshot immuable")
    p_freeze.add_argument("symbol", help=f"un de : {', '.join(UNIVERSE)}")
    p_freeze.add_argument("--tf", default="M15", help="timeframes separes par des virgules")
    p_freeze.add_argument("--side", default="BID,ASK", help="BID, ASK ou les deux")
    p_freeze.add_argument("--start", default="2015-01-01")
    p_freeze.add_argument("--end", default="2026-07-01", help="borne exclue")
    p_freeze.add_argument("--version", dest="version", default="v1")
    p_freeze.set_defaults(func=cmd_freeze)

    p_explore = sub.add_parser(
        "explore", help="teste toutes les familles in-sample et rend le verdict corrige"
    )
    p_explore.add_argument("symbols", nargs="*", help="par defaut : tous les symboles tradables")
    p_explore.add_argument("--tf", default="H1")
    p_explore.add_argument("--controls", type=int, default=N_RANDOM_CONTROL)
    p_explore.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    p_explore.add_argument("--alpha", type=float, default=0.05, help="taux de fausses decouvertes")
    p_explore.add_argument("--out", help="chemin d'un rapport Markdown a ecrire")
    p_explore.add_argument("-v", "--verbose", action="store_true")
    p_explore.set_defaults(func=cmd_explore)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
