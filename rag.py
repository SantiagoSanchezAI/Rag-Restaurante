"""
Asistente RAG para restaurante.

RAG = Retrieval-Augmented Generation.
La idea: en vez de que el modelo invente respuestas, primero BUSCAMOS
información relevante en nuestros documentos (retrieval) y se la pasamos
al modelo para que responda basándose en ella (augmented generation).

Flujo completo:
1. Cargar documentos (menú, info del restaurante)
2. Cortarlos en pedazos (chunks)
3. Convertir cada chunk en un vector de números (embedding)
4. Cuando llega una pregunta, buscar los chunks más parecidos
5. Pasarle esos chunks a Claude para que responda
"""

import os
import glob
import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------

# El cliente de Anthropic lee la API key desde la variable de entorno
# ANTHROPIC_API_KEY. Nunca escribimos la key directo en el código.
client = Anthropic()

# Modelo que convierte texto en embeddings (vectores).
# Este corre localmente y es gratis. Convierte cada frase en 384 números
# que capturan su "significado".
print("Cargando modelo de embeddings (la primera vez puede tardar)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------
# PASO 1 y 2: CARGAR DOCUMENTOS Y CORTARLOS EN CHUNKS
# ---------------------------------------------------------------

def cargar_documentos(carpeta="data"):
    """Lee todos los .txt de la carpeta y los corta en chunks."""
    chunks = []
    # Buscamos todos los archivos .txt en la carpeta data/
    for archivo in glob.glob(os.path.join(carpeta, "*.txt")):
        with open(archivo, "r", encoding="utf-8") as f:
            texto = f.read()
        # Cortamos por párrafos (separados por línea en blanco).
        # Cada párrafo es un "chunk". Es una forma simple de chunking.
        for parrafo in texto.split("\n\n"):
            parrafo = parrafo.strip()
            if parrafo:  # ignoramos los vacíos
                chunks.append(parrafo)
    return chunks


# ---------------------------------------------------------------
# PASO 3: CONVERTIR CHUNKS EN EMBEDDINGS
# ---------------------------------------------------------------

def crear_embeddings(chunks):
    """Convierte cada chunk de texto en un vector de números."""
    # embedder.encode devuelve una lista de vectores, uno por chunk.
    return embedder.encode(chunks)


# ---------------------------------------------------------------
# PASO 4: BUSCAR LOS CHUNKS MÁS RELEVANTES (RETRIEVAL)
# ---------------------------------------------------------------

def buscar_relevantes(pregunta, chunks, embeddings, top_k=3):
    """
    Dada una pregunta, encuentra los top_k chunks más parecidos.
    Usa 'similitud coseno': mide qué tan parecidos son dos vectores.
    """
    # Convertimos la pregunta en un vector, igual que los chunks
    emb_pregunta = embedder.encode([pregunta])[0]

    # Calculamos la similitud entre la pregunta y CADA chunk.
    # Cuanto más alto el número, más parecido.
    similitudes = []
    for emb_chunk in embeddings:
        # Fórmula de similitud coseno
        similitud = np.dot(emb_pregunta, emb_chunk) / (
            np.linalg.norm(emb_pregunta) * np.linalg.norm(emb_chunk)
        )
        similitudes.append(similitud)

    # Ordenamos los chunks de más a menos parecido y tomamos los mejores
    indices_ordenados = np.argsort(similitudes)[::-1][:top_k]
    return [chunks[i] for i in indices_ordenados]


# ---------------------------------------------------------------
# PASO 5: GENERAR LA RESPUESTA CON CLAUDE (GENERATION)
# ---------------------------------------------------------------

def responder(pregunta, chunks_relevantes):
    """Le pasa a Claude la pregunta + el contexto encontrado."""
    # Unimos los chunks relevantes en un solo bloque de contexto
    contexto = "\n\n".join(chunks_relevantes)

    # Este es el prompt: le damos el contexto y le pedimos que responda
    # SOLO con esa información. Así evitamos que invente (alucinaciones).
    prompt = f"""Sos el asistente de un restaurante. Respondé la pregunta del cliente
usando ÚNICAMENTE la información del contexto. Si la respuesta no está en el
contexto, decí que no tenés esa información y sugerí llamar al restaurante.

CONTEXTO:
{contexto}

PREGUNTA DEL CLIENTE:
{pregunta}

RESPUESTA:"""

    # Llamamos a la API de Claude
    respuesta = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return respuesta.content[0].text


# ---------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------

def main():
    print("\nPreparando el asistente...")
    # Cargamos y procesamos los documentos UNA sola vez al inicio
    chunks = cargar_documentos()
    print(f"Documentos cargados: {len(chunks)} chunks.")
    embeddings = crear_embeddings(chunks)
    print("Embeddings listos. Ya podés hacer preguntas.\n")

    print("=" * 50)
    print("ASISTENTE DEL RESTAURANTE LA PARRILLA")
    print("Escribí tu pregunta (o 'salir' para terminar)")
    print("=" * 50)

    while True:
        pregunta = input("\nVos: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break
        if not pregunta:
            continue

        # El flujo RAG completo en 2 líneas:
        relevantes = buscar_relevantes(pregunta, chunks, embeddings)
        respuesta = responder(pregunta, relevantes)
        print(f"\nAsistente: {respuesta}")


if __name__ == "__main__":
    main()
