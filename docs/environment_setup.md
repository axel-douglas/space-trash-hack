# Registro de entorno — Sprint 0

Documento de referencia para reproducir el entorno de entrenamiento y pruebas
de Rex-AI. Incluye dependencias, comandos de verificación y próximos pasos para
automatizar el setup.

## 1. Dependencias críticas

```bash
python --version  # 3.10+
pip install --upgrade pip
pip install -r requirements.txt
# Paquetes opcionales para experimentos avanzados
pip install scikit-learn==1.5.2 xgboost==2.1.1 torch==2.4.1 ax-platform botorch
```

### Versiones verificadas

| Paquete | Versión |
| --- | --- |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| xgboost | 2.1.1 |
| torch | 2.4.1 *(opcional, habilita autoencoder y TabTransformer)* |

Generá `requirements-lock.txt` con `pip freeze` para reproducir exactamente el
entorno en despliegues posteriores.

## 2. Import helpers obligatorios

Todos los scripts CLI y pruebas deben utilizar los helpers de bootstrap para
exponer el paquete `app` sin modificar `sys.path` manualmente.

```python
from app.bootstrap import ensure_project_root
PROJECT_ROOT = ensure_project_root(__file__)
```

En entrypoints de Streamlit utilizá `ensure_streamlit_entrypoint(__file__)`.

## 3. Verificación rápida del pipeline

```bash
python -m app.modules.model_training --help
```

La salida debe incluir los parámetros `--gold`, `--features`, `--samples` y
`--append-logs`. Los warnings de Streamlit (`missing ScriptRunContext`) son
esperados fuera de `streamlit run`.

## 4. Chequeo automatizado recomendado

1. Crear `scripts/check_env.py` que valide imports clave (`pandas`, `sklearn`,
   `app.modules.model_training`).
2. Integrar el script al pipeline de CI para cortar builds cuando falte una
   dependencia.
3. Publicar un snapshot del entorno (Dockerfile o `pip freeze`) en el repositorio
de despliegue para acelerar instalaciones durante el hackathon.

Con este registro cualquier colaborador puede recrear el ambiente base y
proseguir con entrenamiento o demos sin sorpresas.
