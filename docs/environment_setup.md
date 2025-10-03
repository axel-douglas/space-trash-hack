# Registro de entorno — Sprint 0

Este log documenta la preparación mínima del entorno para ejecutar el pipeline de entrenamiento (`app.modules.model_training`).

## 1. Dependencias críticas

Se verificaron/instalaron las librerías base especificadas en `requirements.txt`.

```bash
pip install -r requirements.txt
pip install scikit-learn==1.5.2  # Asegurar disponibilidad
```

### Versiones detectadas

| Paquete | Versión |
| --- | --- |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| xgboost | 2.1.1 |
| torch | *(no instalado; opcional según plan técnico)* |

Se generó además `requirements-lock.txt` mediante `pip freeze` para congelar el entorno actual.

## 2. Verificación del pipeline de entrenamiento

Se ejecutó el módulo en modo ayuda para comprobar que las dependencias permiten importarlo correctamente:

```bash
python -m app.modules.model_training --help
```

Salida relevante:

```
usage: model_training.py [-h] [--gold GOLD] [--features FEATURES] [--samples SAMPLES] [--seed SEED]
                         [--append-logs APPEND_LOGS [APPEND_LOGS ...]]

Training pipeline para Rex-AI
```

Los *warnings* de Streamlit (`missing ScriptRunContext`) son esperados cuando se ejecuta fuera de `streamlit run` y no afectan la carga del módulo.

> 🔁 **Import helper oficial**: evitá añadir rutas manuales en los scripts CLI
> o en las pruebas. Llama a `ensure_project_root(__file__)` antes de importar
> módulos de `app`:
>
> ```python
> from app.bootstrap import ensure_project_root
>
> PROJECT_ROOT = ensure_project_root(__file__)
> ```
>
> Esto mantiene estable la resolución de `app.*` y evita reintroducir hacks
> circulares en `sys.path`.

## 3. Próximos pasos

1. Evaluar si es necesario fijar versiones adicionales (p. ej. `torch==2.4.x`) antes de entrenar modelos TabTransformer.
2. Publicar la imagen o snapshot del entorno en el repositorio de despliegue para repetir la instalación durante el hackathon.
3. Automatizar la verificación mediante un script `scripts/check_env.py` que valide imports y versiones críticas.
