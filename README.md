# 🍽️ Asistente RAG para Restaurante

Un asistente conversacional que responde preguntas sobre un restaurante (menú, precios, horarios, reservas) usando **RAG (Retrieval-Augmented Generation)** con la API de Claude (Anthropic).

A diferencia de un chatbot común, este asistente **no inventa respuestas**: busca la información en documentos reales del restaurante y responde basándose únicamente en ellos. Esto evita las alucinaciones y permite que el negocio actualice su información editando un archivo de texto, sin tocar el código.

> 📝 **Nota:** el restaurante de la carpeta `data/` ("La Parrilla") es **ficticio**. La dirección, los teléfonos, los horarios y el menú se inventaron solo para esta demo.

## 🧠 ¿Cómo funciona?

El sistema implementa el flujo RAG completo:

1. **Carga de documentos** — lee los `.txt` con la información del restaurante.
2. **Chunking** — corta los documentos en fragmentos manejables.
3. **Embeddings** — convierte cada fragmento en un vector numérico que captura su significado (usando `sentence-transformers`, local y gratuito).
4. **Retrieval** — ante una pregunta, calcula la *similitud coseno* para encontrar los fragmentos más relevantes.
5. **Generation** — le pasa esos fragmentos a Claude junto con la pregunta para generar una respuesta precisa y contextualizada.

## 🛠️ Tecnologías

- **Python**
- **Anthropic API** (Claude) — generación de respuestas
- **sentence-transformers** — embeddings locales
- **NumPy** — cálculo de similitud

## 🚀 Cómo usarlo

1. Cloná el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/rag-restaurante.git
   cd rag-restaurante
   ```

2. Instalá las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Configurá tu API key:
   ```bash
   cp .env.example .env
   # Editá .env y poné tu API key de Anthropic
   export ANTHROPIC_API_KEY=tu_api_key
   ```

4. Ejecutá el asistente:
   ```bash
   python rag.py
   ```

## 💬 Ejemplos de preguntas

- "¿Tienen opciones vegetarianas?"
- "¿Cuánto sale el bife de chorizo?"
- "¿Hasta qué hora abren los sábados?"
- "¿Puedo llevar torta para un cumpleaños?"
- "¿Tienen estacionamiento?"

## 📈 Posibles mejoras

- Reemplazar el retrieval manual por una base de datos vectorial (Chroma, FAISS).
- Agregar memoria de conversación para preguntas de seguimiento.
- Conectar con WhatsApp para uso real.
- Interfaz web con FastAPI.

---

Proyecto desarrollado como parte de mi portafolio de IA generativa.
