"""Rendu des resultats d'exploration.

Regle de redaction appliquee ici : un rapport ne doit jamais etre plus affirmatif que
la mesure. Les colonnes brutes sont donc affichees telles quelles, y compris quand
elles sont mauvaises, et le verdict distingue explicitement trois etats — franchit le
gate, survit a la correction multiple, ou rejete.
"""

from __future__ import annotations

import pandas as pd

from alphalab.explore import ExplorationResult

DISPLAY_COLUMNS = [
    "famille",
    "symbole",
    "trades",
    "R_moyen",
    "IC95",
    "taux_reussite",
    "R_cout_majore",
    "bat_hasard",
    "p_value",
    "q_value",
    "gate",
    "survit_correction",
]


def to_console(result: ExplorationResult) -> str:
    """Rendu texte complet, destine au terminal."""
    lines: list[str] = []
    lines.append(f"=== EXPLORATION {result.timeframe} — {', '.join(result.symbols)} ===")
    lines.append("Fenetre in-sample uniquement. Hold-out (>= 2023-01-01) non ouvert.")
    lines.append("Cout : spread reel ASK.open - BID.open, barre par barre.\n")

    if result.table.empty:
        lines.append("Aucune famille evaluee.")
        return "\n".join(lines)

    table = result.table.copy()
    table = table.sort_values("R_moyen", ascending=False, na_position="last")
    with pd.option_context("display.width", 220, "display.max_colwidth", 30):
        lines.append(table[DISPLAY_COLUMNS].to_string(index=False))

    lines.append("\n--- Portes du protocole ---")
    lines.append(
        "Une famille est retenue SSI : >= 100 trades, R > 0 au spread reel, IC excluant 0, "
        "temoin apparie battu dans >= 95 % des cas, R encore > 0 a 1,5x le cout,\n"
        "PUIS survie a la correction pour tests multiples."
    )

    passed = table[table["gate"]]
    lines.append(f"\nFamilles franchissant le gate : {len(passed)} / {len(table)}")
    if not passed.empty:
        lines.append(passed[["famille", "symbole", "R_moyen", "p_value"]].to_string(index=False))

    lines.append("\n--- Verdict ---")
    lines.append(result.verdict())

    if not passed.empty and result.survivors.empty:
        lines.append(
            "\nLecture : des familles franchissent les portes prises isolement, mais aucune "
            "ne resiste au nombre d'essais menes. C'est le comportement attendu quand on "
            "explore largement sans edge reel — et exactement ce que la correction sert a "
            "reveler."
        )

    lines.append("\n--- Reserves ---")
    lines.append(
        "- Resultats in-sample : ils ne disent rien de la performance future. Seul le "
        "hold-out, ouvert une seule fois, le fera."
    )
    ambiguous = table["taux_ambigu"].max() if "taux_ambigu" in table else 0
    lines.append(
        f"- Taux de barres ambigues (stop et objectif atteignables) le plus eleve : "
        f"{ambiguous:.1%}. Au-dela de quelques pourcents, le resultat depend fortement de "
        "l'hypothese intrabarre."
    )
    if result.missing_peers:
        lines.append(
            f"- Facteur dollar indisponible ({', '.join(result.missing_peers)} absents) : "
            "les familles inter-actifs n'ont teste qu'un axe risk-on/risk-off."
        )
    return "\n".join(lines)


def _markdown_table(table: pd.DataFrame) -> str:
    """Table Markdown sans dependance externe.

    `DataFrame.to_markdown` exige `tabulate`, une dependance de plus pour une mise en
    forme de dix lignes. Le rapport est un livrable du projet : il ne doit pas dependre
    d'un paquet optionnel.
    """
    columns = list(table.columns)
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def to_markdown(result: ExplorationResult) -> str:
    """Rendu Markdown, destine a un compte-rendu de session."""
    lines: list[str] = []
    lines.append(f"# Exploration {result.timeframe} — {', '.join(result.symbols)}\n")
    lines.append(
        "Fenetre in-sample, cout d'execution reel (spread ASK−BID barre par barre). "
        "Hold-out (≥ 2023-01-01) non ouvert.\n"
    )
    if result.table.empty:
        lines.append("_Aucune famille evaluee._")
        return "\n".join(lines)

    table = result.table.copy().sort_values("R_moyen", ascending=False, na_position="last")
    lines.append(_markdown_table(table[DISPLAY_COLUMNS]))
    lines.append("\n## Verdict\n")
    lines.append(result.verdict())
    lines.append("\n## Reserves\n")
    lines.append(
        "- Resultats in-sample : ils ne disent rien de la performance future.\n"
        f"- Taux de barres ambigues le plus eleve : {table['taux_ambigu'].max():.1%}.\n"
        "- Une q-value faible ne suffit pas : la correction ne juge que la p-value du "
        "temoin, pas le signe de l'esperance. Seule la colonne `survit_correction` "
        "combine les deux."
    )
    return "\n".join(lines)
