"""Implementações concretas de :class:`~diabetes_risk.domain.protocols.DataSource`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class CsvDataSource:
    """Lê os dados brutos de um arquivo CSV local.

    Args:
        path: Caminho do CSV.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> pd.DataFrame:
        """Lê o CSV.

        Raises:
            FileNotFoundError: Se o arquivo não existir, com instrução de download.
        """
        if not self._path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em '{self._path}'. Baixe-o do Kaggle "
                "('Diabetes Health Indicators Dataset') e salve nesse caminho."
            )
        return pd.read_csv(self._path)
