"""Etapa ``prepare``: CSV bruto -> treino/teste validados em Parquet."""

from __future__ import annotations

import logging

from diabetes_risk.config import PipelineConfig
from diabetes_risk.data.preprocessing import (
    DataSplit,
    build_binary_target,
    drop_duplicate_rows,
    split_train_test,
    validate_raw,
)
from diabetes_risk.data.sources import CsvDataSource
from diabetes_risk.domain.protocols import DataSource
from diabetes_risk.domain.schemas import BINARY_TARGET

logger = logging.getLogger(__name__)


def run_prepare(config: PipelineConfig, source: DataSource | None = None) -> DataSplit:
    """Valida, limpa e divide os dados brutos.

    Passos: validação de schema -> alvo binário -> deduplicação -> split
    estratificado -> gravação em ``config.data.processed_dir``.

    Args:
        config: Configuração do pipeline.
        source: Fonte dos dados; por padrão, o CSV de ``config.data.raw_path``.

    Returns:
        Os conjuntos de treino e teste gravados.
    """
    data_cfg = config.data
    source = source or CsvDataSource(data_cfg.raw_path)

    raw = validate_raw(source.load())
    logger.info("Dados brutos validados: %d linhas", len(raw))

    dataset = build_binary_target(raw, data_cfg.positive_classes)
    if data_cfg.drop_duplicates:
        before = len(dataset)
        dataset = drop_duplicate_rows(dataset)
        logger.info("Duplicatas removidas: %d", before - len(dataset))

    split = split_train_test(dataset, data_cfg.test_size, data_cfg.random_state)

    data_cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_parquet(data_cfg.train_path, index=False)
    split.test.to_parquet(data_cfg.test_path, index=False)
    logger.info(
        "Treino: %d linhas | Teste: %d linhas | Positivos no treino: %.1f%%",
        len(split.train),
        len(split.test),
        100 * split.train[BINARY_TARGET].mean(),
    )
    return split
