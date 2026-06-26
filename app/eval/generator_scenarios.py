import json
import random
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import SentenceTransformer, util

# Cargar un modelo de embeddings ligero y preciso
model_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Definir modificadores para forzar variedad
perfiles = ["estudiante mochilero", "ejecutivo estresado", "jubilado aventurero", "familia numerosa", "nómada digital"]
destinos = ["Uzbekistán", "Kirguistán", "Bután", "Namibia", "Bolivia", "Albania", "Laos"]
tonos = ["muy formal y técnico", "coloquial y emocionado", "directo y urgente", "con muchas dudas y preguntas"]

def get_random_context():
    return f"Perfil: {random.choice(perfiles)}. Destino principal: {random.choice(destinos)}. Tono de voz: {random.choice(tonos)}."

def es_diverso(nuevo_texto, corpus_existente):
    if not corpus_existente: return True
    
    embeddings_nuevo = model_embeddings.encode([nuevo_texto])
    embeddings_corpus = model_embeddings.encode(corpus_existente)
    
    similitudes = util.cos_sim(embeddings_nuevo, embeddings_corpus)
    return all(s < 0.4 for s in similitudes[0])

# 1. Configurar tu modelo local (Temperatura media-alta para garantizar diversidad)
model = ChatOpenAI(
    model="qwen2.5:0.5b", 
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    temperature=0.4 # Bajamos un poco la temperatura para evitar que rompa el JSON
)

parser = JsonOutputParser()

# 2. Plantilla MUY estricta para forzar el formato en modelos pequeños
prompt = ChatPromptTemplate.from_template(
    "Actúa como un diseñador de corpus de evaluación. "
    "Genera UN escenario sintético en formato JSON para la intención: {intent}. "
    "DEBE respetar este contexto obligatorio: {contexto}. "
    "REGLAS ESTRICTAS PARA EL JSON:\n"
    "1. NO uses comentarios como // o #.\n"
    "2. Usa EXACTAMENTE esta estructura de claves:\n"
    "{{\n"
    "  \"intent\": \"{intent}\",\n"
    "  \"user_profile\": \"...\",\n"
    "  \"sessions\": [\n"
    "    {{\"time\": \"Día 1\", \"user_utterance\": \"...\", \"expected_agent_action\": \"...\"}},\n"
    "    {{\"time\": \"Día 3\", \"user_utterance\": \"...\", \"expected_agent_action\": \"...\"}},\n"
    "    {{\"time\": \"Día 7\", \"user_utterance\": \"...\", \"expected_agent_action\": \"...\"}}\n"
    "  ]\n"
    "}}\n"
    "DEVUELVE ÚNICAMENTE EL JSON. SIN MARKDOWN."
)

chain = prompt | model | parser

# 3. Lista de tus 10 intenciones
intents = [
    "Planificación con Restricciones Estrictas",
    "Modificación Crítica de Reserva",
    "Resolución de Problemas en Destino (Urgencia)",
    "Consulta Compleja de Políticas/Legal",
    "Adición de Servicios Auxiliares (Upselling)",
    "Coordinación de Viajes Grupales/Multi-Origen",
    "Gestión de Reembolsos y Quejas",
    "Asesoramiento Basado en Clima/Temporalidad",
    "Fidelización y Viajes Recurrentes",
    "Interrupciones y Cambio de Contexto Abrupto"
]

corpus_final = []
id_counter = 4 # Empezamos en 4 (asumiendo que ya tienes 3)

# 4. Bucle de generación
for intent in intents:
    print(f"\n--- Generando escenarios para: {intent} ---")
    for i in range(5): 
        print(f"Generando escenario {id_counter}...")
        diversidad_lograda = False
        intentos = 0
        
        while not diversidad_lograda and intentos < 5: # Máximo 5 intentos por escenario para no hacer bucle infinito
            contexto = get_random_context()
            
            try:
                # 1. Llamada al LLM
                resultado = chain.invoke({"intent": intent, "contexto": contexto})
                
                # 2. Extraer texto para evaluar
                texto_escenario = " ".join([s.get('user_utterance', '') for s in resultado.get('sessions', [])])
                
                # 3. Verificamos si es diverso
                if es_diverso(texto_escenario, [item.get('texto_plano', '') for item in corpus_final]):
                    resultado['scenario_id'] = f"TFM_{id_counter:03d}"
                    resultado['contexto_usado'] = contexto # <-- Aquí agregamos el contexto usado
                    resultado['texto_plano'] = texto_escenario 
                    
                    corpus_final.append(resultado)
                    diversidad_lograda = True
                    id_counter += 1
                    print(f"✅ Escenario {id_counter-1} generado con éxito.")
                else:
                    print(f"⚠️ Escenario muy similar detectado, regenerando...")
                    intentos += 1
            except Exception as e:
                # Si el modelo alucina un JSON inválido o pone comentarios, cae aquí y lo vuelve a intentar
                print(f"❌ Error de formato JSON devuelto por el modelo. Reintentando... ({intentos+1}/5)")
                intentos += 1
                
        if not diversidad_lograda:
            print(f"🚨 No se pudo generar un escenario diverso/válido tras 5 intentos. Saltando...")

# 5. Limpiar el 'texto_plano' antes de guardar si lo deseas (opcional)
for item in corpus_final:
    item.pop('texto_plano', None)

# 6. Guardar en disco
with open("corpus_evaluacion_tfm.json", "w", encoding="utf-8") as f:
    json.dump(corpus_final, f, indent=2, ensure_ascii=False)

print(f"\n¡Generación completada! Se guardaron {len(corpus_final)} escenarios.")