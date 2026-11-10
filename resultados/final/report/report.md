# Resultados de la evaluación

Fuente del veredicto: **juez LLM (deepseek-v4-pro)**. Filas: 3000; sin veredicto o pendientes: 0.

## Exactitud de recuerdo

Exactitud (%) por categoría con historial de 20 sesiones, voto mayoritario de 3 corrida(s) por caso.

| Categoría | C0 Stateless | C1 Contexto completo | C2 RAG simple | C3 Memori | C4 Oracle |
| --- | --- | --- | --- | --- | --- |
| Extracción | 0 (n=20) | 100 (n=20) | 45 (n=20) | 85 (n=20) | 100 (n=20) |
| Multi-sesión | 0 (n=20) | 90 (n=20) | 5 (n=20) | 90 (n=20) | 95 (n=20) |
| Temporal | 0 (n=20) | 90 (n=20) | 35 (n=20) | 75 (n=20) | 100 (n=20) |
| Actualización | 0 (n=20) | 100 (n=20) | 20 (n=20) | 90 (n=20) | 100 (n=20) |
| Abstención | 100 (n=20) | 100 (n=20) | 100 (n=20) | 95 (n=20) | 100 (n=20) |
| **Global** | **20.0** | **96.0** | **41.0** | **87.0** | **99.0** |

**Variación entre corridas** (exactitud global por corrida, media ± DE):

| Condición | Corrida 1 | Corrida 2 | Corrida 3 | Media ± DE |
| --- | --- | --- | --- | --- |
| C0 | 20.0 | 20.0 | 20.0 | 20.0 ± 0.0 |
| C1 | 97.0 | 97.0 | 95.0 | 96.3 ± 1.2 |
| C2 | 40.0 | 41.0 | 41.0 | 40.7 ± 0.6 |
| C3 | 86.0 | 87.0 | 89.0 | 87.3 ± 1.5 |
| C4 | 98.0 | 99.0 | 96.0 | 97.7 ± 1.5 |

Comparaciones pareadas de C3 con 20 sesiones (McNemar exacto, bootstrap pareado, Holm).

| Comparación | n | Diferencia (pp) | IC 95 % | p | p ajustado (Holm) |
| --- | --- | --- | --- | --- | --- |
| C3 contra C0 | 100 | +67.0 | [+57.0, +76.0] | < 0.001 | < 0.001 |
| C3 contra C2 | 100 | +46.0 | [+35.0, +57.0] | < 0.001 | < 0.001 |
| C3 contra C1 | 100 | -9.0 | [-16.0, -3.0] | 0.022 | 0.022 |

**H2** (multi-sesión y actualización, C3 contra C0): +90.0 pp, n = 40.

**H3** (C3 contra C1 por longitud; margen de equivalencia ±5 pp):

| Sesiones | Diferencia (pp) | IC 95 % | p | Lectura |
| --- | --- | --- | --- | --- |
| 5 | -8.0 | [-14.0, -2.0] | 0.021 | Ventaja de C1 |
| 20 | -9.0 | [-16.0, -3.0] | 0.022 | Ventaja de C1 |

**Recall@5 de memoria** (casos no de abstención, voto mayoritario):

| Sesiones | C2 | C3 |
| --- | --- | --- |
| 5 | 37.5 (n=80) | 78.8 (n=80) |
| 20 | 25.0 (n=80) | 78.8 (n=80) |

**Tokens de entrada por pregunta** (media; en C3, coste de escritura de memoria por turno sembrado aparte):

| Sesiones | C0 | C1 | C2 | C3 | C4 | C3 escritura/turno |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 096 | 3 067 | 1 348 | 1 411 | 1 639 | 307 |
| 20 | 1 096 | 9 349 | 1 359 | 1 411 | 1 639 | 308 |

## Repetición y pasos para la resolución

Usuario simulado, 60 tareas, historial de 20 sesiones, 3 corrida(s).

| Métrica | C0 | C3 | Reducción de C3 frente a C0 |
| --- | --- | --- | --- |
| Tasa de repetición (RR, %) | 30.8 | 11.0 | 65.7 % relativa, IC 95 % [52, 77] |
| Pasos para la resolución (STR, media) | 5.1 | 3.7 | 27.9 % relativa |
| Tareas completadas (%) | 87 | 96 | |

Las conversaciones no completadas cuentan con el máximo de turnos (censuradas).

## Latencia y costo

**Variante offload.** Latencia por etapa con Memori, 200 peticiones por nivel, 10 usuarios con 20 sesiones (MEMORI_OFFLOAD_SYNC=True).

| Concurrencia | Recuperación p95 | Generación p95 | Escritura p95 | Total p95 | Errores | Fallos de memoria | ¿Recuperación p95 < 300 ms? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40 | 2857 | 2008 | 4539 | 0 | 0 | Sí |
| 5 | 48 | 3663 | 1543 | 5596 | 0 | 0 | Sí |
| 10 | 87 | 2637 | 334 | 3313 | 0 | 0 | Sí |
| 20 | 208 | 4959 | 750 | 5737 | 0 | 0 | Sí |

**Variante sync.** Latencia por etapa con Memori, 200 peticiones por nivel, 10 usuarios con 20 sesiones (MEMORI_OFFLOAD_SYNC=False).

| Concurrencia | Recuperación p95 | Generación p95 | Escritura p95 | Total p95 | Errores | Fallos de memoria | ¿Recuperación p95 < 300 ms? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 45 | 4310 | 2446 | 5929 | 0 | 0 | Sí |
| 5 | 45 | 4229 | 2315 | 5536 | 0 | 0 | Sí |
| 10 | 35 | 5473 | 2973 | 7311 | 0 | 0 | Sí |
| 20 | 27 | 9570 | 5376 | 13500 | 0 | 0 | Sí |

## Validación externa (LongMemEval-S)

Exactitud (%) en el subconjunto de LongMemEval-S, una corrida, n = 60 preguntas.

| Medida | C0 Stateless | C1 Contexto completo | C3 Memori | C4 Oracle |
| --- | --- | --- | --- | --- |
| Exactitud (%) | 21.7 | 73.3 | 65.0 | 71.7 |

C3 contra C1: -8.3 pp, n = 60.

## Estado de los objetivos

Trazabilidad entre objetivos, resultados y estado.

| Objetivo o criterio | Resultado | Estado |
| --- | --- | --- |
| Exactitud mayor que stateless (H2 global) | +67.0 pp (p Holm < 0.001) | Cumplido |
| Ventaja frente a contexto completo (H3) | 5 sesiones: -8.0 pp (ventaja de C1); 20 sesiones: -9.0 pp (ventaja de C1) | No cumplido |
| Reducción de RR ≥ 25 % (H1) | 65.7 % | Cumplido (dentro o por encima del valor esperado) |
| Reducción de STR ≥ 25 % | 27.9 % | Cumplido |
| Recall@5 ≥ 90 % | 78.8 % (20 sesiones) | No cumplido |
| Latencia p95 de recuperación < 300 ms (H4) | offload: cumple en concurrencia 1, 5, 10, 20; sync: cumple en concurrencia 1, 5, 10, 20 | Cumplido |
| Similitud entre escenarios < 0.4 | Máximo 0.39 (tfidf); 0 pares por encima | Cumplido |
| Acuerdo juez-humano ≥ 90 % | deepseek:deepseek-v4-pro: 100.0 % (κ = 1.00, n = 49) | Cumplido |
| Validación externa, LongMemEval-S (60 preguntas) | C0 21.7 %, C1 73.3 %, C3 65.0 %, C4 71.7 % | Orientativa: RMM reporta 70.4 % con otro recuperador y otro juez (Tan, Z. et al., 2025) |
