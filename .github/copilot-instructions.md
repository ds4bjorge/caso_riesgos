- Me responderás siempre en español de España.
- Comenzarás cada mensaje con este emoji 🤖
- Siempre genera código para el script abierto, nunca para la ventana interactiva, salvo que se indique explícitamente lo contrario.
- La ventana interactiva solo se usará para verificar resultados o mensajes de error, pero no para generar código.
- No expliques el código si no se solicita.


## ESTADO ACTUAL DEL PROYECTO

**Fase completada**: A_11_API_Deployer
**Modo**: LOCAL + RENDER_ROOT_READY

**Rutas clave**:
	- Motor scoring: `07_despliegue/api/scoring.py`
	- Artefacto: `07_despliegue/api/artefacto_pipeline.pkl`
	- API local: `07_despliegue/api/main.py`

**Comando de arranque local validado:**
```bash
uvicorn api.main:app --reload --app-dir ..
```

**Contrato final de entrada y salida:**
	- Endpoint canónico de inferencia: `POST /predict`
	- Formato canónico de entrada: lista JSON de registros `[{...}]`
	- Formato canónico de salida: lista JSON de objetos serializada desde el DataFrame del motor
	- Modalidad de salida elegida: salida completa
	- Columnas reales de salida: todas las columnas devueltas por `scoring_df(df)`
	- Ejemplo válido de respuesta final:
```json
[
	{
		"id_cliente": 1,
		"principal": 10000.0,
		"tipo_interes": 4.5,
		"num_cuotas": " 36 months",
		"finalidad": "home_improvement",
		"vivienda": "MORTGAGE",
		"score": 0.83
	}
]
```

**Configuración validada de Render:**
	- Root Directory: 07_despliegue/api
	- Build Command: pip install -r requirements.txt
	- Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT --app-dir ..
	- Key: PYTHON_VERSION
	- Value: 3.11.15

**Siguiente paso recomendado:**
	- Commit y push a GitHub incluyendo `07_despliegue/api/data/demo_03_riesgos.csv`
	- Alta manual del servicio en Render usando la configuración anterior
