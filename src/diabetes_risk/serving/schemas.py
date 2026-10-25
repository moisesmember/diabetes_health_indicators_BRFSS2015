"""Contratos de entrada e saída da API (Pydantic).

As faixas de cada campo espelham ``FEATURE_RANGES`` do domínio; um teste
garante que as duas definições nunca divirjam.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from diabetes_risk.models.risk import Classification, RiskLevel


class PatientFeatures(BaseModel):
    """Indicadores de saúde de um paciente, no formato do BRFSS 2015."""

    model_config = ConfigDict(extra="forbid")

    HighBP: float = Field(ge=0, le=1, description="Pressão alta (0 = não, 1 = sim)")
    HighChol: float = Field(ge=0, le=1, description="Colesterol alto (0 = não, 1 = sim)")
    CholCheck: float = Field(ge=0, le=1, description="Checou colesterol nos últimos 5 anos")
    BMI: float = Field(ge=12, le=98, description="Índice de massa corporal")
    Smoker: float = Field(ge=0, le=1, description="Fumou ao menos 100 cigarros na vida")
    Stroke: float = Field(ge=0, le=1, description="Já teve AVC")
    HeartDiseaseorAttack: float = Field(ge=0, le=1, description="Doença coronariana ou infarto")
    PhysActivity: float = Field(ge=0, le=1, description="Atividade física nos últimos 30 dias")
    Fruits: float = Field(ge=0, le=1, description="Come frutas 1+ vez ao dia")
    Veggies: float = Field(ge=0, le=1, description="Come vegetais 1+ vez ao dia")
    HvyAlcoholConsump: float = Field(ge=0, le=1, description="Consumo pesado de álcool")
    AnyHealthcare: float = Field(ge=0, le=1, description="Possui plano/cobertura de saúde")
    NoDocbcCost: float = Field(ge=0, le=1, description="Deixou de ir ao médico pelo custo")
    GenHlth: float = Field(ge=1, le=5, description="Saúde geral (1 = excelente ... 5 = ruim)")
    MentHlth: float = Field(ge=0, le=30, description="Dias de saúde mental ruim (últimos 30)")
    PhysHlth: float = Field(ge=0, le=30, description="Dias de saúde física ruim (últimos 30)")
    DiffWalk: float = Field(ge=0, le=1, description="Dificuldade para andar ou subir escadas")
    Sex: float = Field(ge=0, le=1, description="Sexo (0 = feminino, 1 = masculino)")
    Age: float = Field(ge=1, le=13, description="Faixa etária (1 = 18-24 ... 13 = 80+)")
    Education: float = Field(ge=1, le=6, description="Escolaridade (1 a 6)")
    Income: float = Field(ge=1, le=8, description="Faixa de renda (1 a 8)")


class PredictionRequest(BaseModel):
    """Lote de pacientes a pontuar."""

    patients: list[PatientFeatures] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    """Resultado da triagem de um paciente."""

    probability: float = Field(
        description="Score do modelo (0 a 1). Ordena o risco, mas não é uma probabilidade "
        "calibrada; para interpretar, use risk_level e band_prevalence."
    )
    at_risk: bool = Field(description="True se o score atinge o limiar de decisão do modelo")
    classification: Classification = Field(
        description="Indicativo de triagem: 'com diabetes' (inclui pré-diabetes) ou 'sem diabetes'"
    )
    risk_level: RiskLevel = Field(
        description="Chance de diabetes ou pré-diabetes: alta, média ou baixa"
    )
    band_prevalence: float | None = Field(
        description="Nos dados de treino, fração de pessoas desta mesma faixa que tinham "
        "diabetes ou pré-diabetes (ex.: 0.5 = 50 em cada 100)"
    )
    recommendation: str = Field(description="Indicação do que fazer a seguir")
    requires_confirmation: bool = Field(
        description="True quando o resultado precisa ser confirmado por exame laboratorial"
    )
    confirmatory_tests: list[str] = Field(description="Exames indicados para a confirmação")


class PredictionResponse(BaseModel):
    """Resultado do lote, na mesma ordem da requisição."""

    model: str
    threshold: float
    disclaimer: str = Field(description="Aviso: triagem estatística, não diagnóstico")
    predictions: list[Prediction]


class RiskBandsInfo(BaseModel):
    """Cortes de score que definem cada faixa de chance."""

    medium_from: float = Field(description="Score a partir do qual a chance é média")
    high_from: float = Field(description="Score a partir do qual a chance é alta")
    prevalence: dict[RiskLevel, float | None] = Field(
        description="Fração com diabetes/pré-diabetes em cada faixa (dados de treino)"
    )


class ModelInfo(BaseModel):
    """Metadados do modelo em serviço."""

    model: str
    threshold: float
    trained_at: str
    risk_bands: RiskBandsInfo
