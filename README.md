# Sistema de apoyo para detección temprana de riesgo cardíaco

## Descripción

Este proyecto implementa un sistema de aprendizaje automático para clasificar presencia o ausencia de enfermedad cardíaca usando el dataset UCI Heart Disease.

## Dataset

Se utiliza principalmente el archivo `processed.cleveland.data` de UCI Heart Disease.

La variable objetivo original `num` se transforma a clasificación binaria:

- 0: ausencia de enfermedad cardíaca
- 1: presencia de enfermedad cardíaca

También se incluye como comparación el dataset pequeño `problemas_del_corazon.csv`.

## Estructura del proyecto

```text
data/raw/       Datos originales descargados
data/processed/ Datos procesados generados por el código
notebooks/      Análisis exploratorio y experimentos
src/            Código fuente
results/        Tablas, figuras y modelo final
report/         Informe final