# Proyecto integrador — Visualización de Datos
## US Flight Delays 2025 · BTS On-Time Performance

### 1. Problema y usuario
**Problema.** La puntualidad del transporte aéreo doméstico de Estados Unidos cambia según mes, franja horaria, aerolínea, aeropuerto, estado y causa de demora. Analizar solo promedios globales puede ocultar puntos operativos de alto riesgo.

**Usuario objetivo.** Analista de operaciones / planificación de red de una aerolínea o administrador aeroportuario que necesita priorizar acciones para reducir demoras, cancelaciones y disrupciones.

**Decisiones que la solución debe apoyar (5).**
1. Priorizar meses, días y franjas horarias que requieren mayor capacidad o contingencia.
2. Identificar aerolíneas y aeropuertos con mayor exposición a demoras, controlando por volumen.
3. Identificar estados de origen con concentración geográfica de riesgo.
4. Distinguir qué causas explican los minutos de demora: aerolínea, clima, NAS, seguridad o llegada tardía de aeronave.
5. Definir acciones operativas basadas en puntualidad, cancelación, desvío y severidad de demora.

### 2. Preguntas analíticas
1. ¿Cómo evoluciona el volumen de vuelos por mes durante 2025?
2. ¿Cómo cambia la tasa de puntualidad de llegada a lo largo del año?
3. ¿Qué meses presentan simultáneamente baja puntualidad y alta cancelación?
4. ¿Qué días de la semana y franjas de salida presentan mayor probabilidad de demora?
5. ¿Qué aerolíneas muestran mayor tasa de demora y con qué nivel de incertidumbre estadística?
6. ¿Qué aeropuertos combinan alto volumen, alta demora y alta cancelación?
7. ¿En qué estados de origen se concentra el mayor porcentaje de vuelos demorados?
8. ¿Cuál es la contribución relativa de cada causa a los minutos totales de demora?
9. ¿Existe relación entre la distancia del vuelo y la tasa de demora?
10. ¿Cuál es la severidad de las demoras mediante promedio y percentil 95?

### 3. Matriz pregunta–variable–tarea–gráfico
| Pregunta | Variables principales | Tarea analítica | Visualización |
|---|---|---|---|
| P1 | Month, Flights | Tendencia / volumen | Barras mensuales |
| P2 | Month, ArrDel15 | Tendencia / comparación | Línea de puntualidad |
| P3 | Month, ArrDel15, Cancelled | Detección de periodos críticos | Línea doble + volumen |
| P4 | DayOfWeek, DepTimeBlk, ArrDel15 | Patrón temporal | Barras + línea por franja |
| P5 | Reporting_Airline, ArrDel15 | Ranking con incertidumbre | Barras horizontales + IC 95% |
| P6 | Origin, ArrDel15, Cancelled, Flights | Riesgo vs volumen | Dispersión con tamaño/color |
| P7 | OriginState, ArrDel15 | Distribución geográfica | Coroplético USA |
| P8 | *Delay | Composición | Barras por causa |
| P9 | Distance, ArrDel15 | Asociación | Dispersión por bandas |
| P10 | ArrDelayMinutes | Severidad | KPI P95 + promedio |

### 4. Diccionario de KPI
| KPI | Fórmula | Unidad | Frecuencia | Dirección deseada | Referencia |
|---|---|---|---|---|---|
| Vuelos programados | Σ Flights | vuelos | mensual | contexto | — |
| Puntualidad de llegada | 1 − AVG(ArrDel15) | % | mensual | ↑ | ≥ 80% como meta operativa configurable |
| Tasa de demora | AVG(ArrDel15) | % | mensual | ↓ | < 20% |
| Tasa de cancelación | AVG(Cancelled) | % | mensual | ↓ | < 2% |
| Tasa de desvío | AVG(Diverted) | % | mensual | ↓ | lo menor posible |
| Demora media en vuelos retrasados | AVG(ArrDelayMinutes) donde ArrDel15=1 | minutos | mensual | ↓ | — |
| Demora P95 | Q95(ArrDelayMinutes) | minutos | mensual | ↓ | — |
| Tasa de demora de salida | AVG(DepDel15) | % | mensual | ↓ | < 20% |

> Nota: las metas son referencias de gestión configurables; el dashboard muestra valores observados y no las interpreta como normas regulatorias.

### 5. Modelo de datos
**Granularidad de la tabla de hechos:** un vuelo / segmento doméstico reportado.

**Hechos / medidas:** Flights, DepDelayMinutes, ArrDelayMinutes, ArrDel15, DepDel15, Cancelled, Diverted, Distance, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay.

**Dimensiones:** Fecha (año, mes, día, día de semana), Aerolínea, Aeropuerto de origen, Aeropuerto de destino, Estado, Franja horaria y Causa.

**Modelo recomendado:** esquema estrella conceptual con `FactFlight` al centro y dimensiones `DimDate`, `DimCarrier`, `DimOrigin`, `DimDestination`, `DimTimeBlock`.

### 6. Auditoría visual
- Se evita usar gráficos 3D y ejes truncados que exageren diferencias.
- El color rojo se reserva para alertas/cancelación; azul y verde azulado para métricas neutrales/positivas.
- Los rankings se acompañan de volumen mínimo para evitar conclusiones por muestras pequeñas.
- La comparación de aerolíneas incorpora IC 95% Wilson para representar incertidumbre.
- El mapa representa proporciones por estado, no recuentos absolutos, para evitar confundir volumen con riesgo.
- Todos los gráficos incluyen unidad, contexto y filtros activos.

### 7. Wireframe — 4 vistas
**Vista 1 · Resumen ejecutivo:** 6 KPI + tendencia mensual + lectura ejecutiva.  
**Vista 2 · Puntualidad y tiempo:** día de semana + franja de salida + distancia.  
**Vista 3 · Aerolíneas y aeropuertos:** aerolíneas con IC 95% + scatter volumen/riesgo.  
**Vista 4 · Geografía y causas:** mapa por estado + causas de demora + decisiones apoyadas.

### 8. Implementación Hito 2
La app se implementa en **Python + Streamlit + Plotly + DuckDB**, descarga el dataset público desde Kaggle con `kagglehub`, crea un parquet optimizado y ofrece filtros de mes, aerolínea, estado y volumen mínimo de aeropuerto. Incluye más de cinco visualizaciones, análisis temporal, componente geoespacial, incertidumbre y cuatro vistas interactivas.

### 9. Reproducibilidad
1. Abrir `colab_flight_delays_2025.ipynb` en Google Colab.
2. Ejecutar las celdas en orden.
3. La primera ejecución instala librerías y descarga el dataset de Kaggle.
4. La app prepara un parquet optimizado para acelerar las consultas.
5. Se inicia Streamlit y se genera una URL temporal con Cloudflare Quick Tunnel.
6. Para una ejecución local: `pip install -r requirements.txt` y `streamlit run app.py`.

### 10. Uso de IA — registro crítico sugerido
La IA se utilizó para proponer estructura de código, alternativas de visualización y documentación. Las definiciones de campos/KPI deben verificarse contra la documentación oficial BTS y los resultados numéricos se calculan directamente desde los datos mediante código reproducible.
