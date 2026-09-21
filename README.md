# US Flight Delays 2025 — Dashboard Web App

Proyecto de Ciencia de Datos y Visualización desarrollado a partir del dataset público **US Flight Delays 2025 (BTS On-Time Performance)** de Kaggle y alineado con las páginas 1–3 de la guía del Proyecto Integrador.

## Archivos
- `app.py`: dashboard Streamlit completo, 4 vistas, filtros, mapa, KPI e incertidumbre.
- `colab_flight_delays_2025.ipynb`: notebook listo para Google Colab; instala, genera la app y la publica mediante un túnel temporal.
- `colab_flight_delays_2025.py`: versión Python del flujo de Colab.
- `dashboard_vista_previa.html`: dashboard HTML autónomo de presentación y vista previa.
- `HITO_1_Y_2.md`: ficha del problema, usuario, decisiones, 10 preguntas, matriz analítica, 8 KPI, modelo, auditoría, wireframe y reproducibilidad.
- `requirements.txt`: dependencias.

## Ejecución local
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Ejecución en Google Colab
Abre `colab_flight_delays_2025.ipynb`, selecciona **Entorno de ejecución > Ejecutar todo** y espera la URL `trycloudflare.com`.

## Datos
La app usa `kagglehub.dataset_download("a7madmostafa/us-flight-delays-2025-bts-on-time-performance")`. No se redistribuye el dataset dentro del ZIP: se descarga desde su fuente en cada entorno.

## Diseño
Paleta profesional azul marino / azul / teal con ámbar y rojo solo para énfasis. La app evita gráficos 3D, usa escala consistente, incluye filtros y aporta incertidumbre mediante intervalos Wilson del 95%.
