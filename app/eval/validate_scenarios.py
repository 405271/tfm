import os
import json
import glob
from sentence_transformers import SentenceTransformer, util

# Cargar un modelo de embeddings ligero y preciso
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

escenarios_textos = []
escenarios_ids = []

# Obtiene la ruta absoluta de donde está el archivo .py actual
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Busca los json en esa ruta específica
archivos_json = glob.glob(os.path.join(directorio_actual, "*.json"))

print(f"Buscando en: {directorio_actual}")

# 2. Buscar todos los archivos JSON en la carpeta actual
archivos_json = glob.glob(os.path.join(directorio_actual, "*.json"))
print(f"Archivos encontrados: {archivos_json}")

# 3. Leer cada archivo y extraer las expresiones del usuario
for archivo in archivos_json:
    print(f"Archivo encontrado: {os.path.basename(archivo)}")
    with open(archivo, "r", encoding="utf-8") as f:
        datos = json.load(f)
        
        # Asumiendo que el JSON tiene una lista de escenarios
        for escenario in datos:
            scenario_id = escenario.get("scenario_id", f"Desconocido_en_{archivo}")
            
            # Concatenar todos los user_utterance de las 3 sesiones del escenario
            utterances = [
                sesion.get("user_utterance", "") 
                for sesion in escenario.get("sessions", [])
            ]
            texto_completo = " ".join(utterances)
            
            escenarios_ids.append(scenario_id)
            escenarios_textos.append(texto_completo)

print(f"Total de escenarios cargados para evaluar: {len(escenarios_textos)}")

# 4. Calcular los embeddings para todos los escenarios
print("Calculando matriz de similitud...")
# UN SOLO ENCODE para todos los textos a la vez (es miles de veces más rápido)
embeddings = model.encode(escenarios_textos, batch_size=32)
matriz_similitud = util.cos_sim(embeddings, embeddings)

# 5. Validar que ningún par de escenarios distintos supere el 0.4 de similitud
exito = True
limite_similitud = 0.4

for i in range(len(escenarios_textos)):
    for j in range(i + 1, len(escenarios_textos)):
        similitud = matriz_similitud[i][j].item()
        
        if similitud >= limite_similitud:
            print(f"⚠️ Alerta: '{escenarios_ids[i]}' y '{escenarios_ids[j]}' son muy similares (Similitud: {similitud:.2f})")
            exito = False

if exito:
    print(f"✅ Corpus validado: Alta diversidad semántica (Similitud inter-escenario < {limite_similitud} garantizada).")
else:
    print("❌ Existen escenarios demasiado similares. Debes regenerar los marcados con alerta.")