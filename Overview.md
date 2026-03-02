# Architecture: AI-Powered Course Generation Pipeline

This architecture breaks the process down into five distinct phases: Data Ingestion, High-Level Planning, Detail Generation, Media Generation, and Delivery.

## Phase 1: Data Ingestion & Preprocessing (The Knowledge Base)
Before the LLM can generate a plan, it needs to understand the material in a structured way.

* **Document Parsers:** Extract text from PDFs, slides, and text files. You might need OCR (like Tesseract) for scanned previous exams.
* **Chunking Strategy:** Split the raw text into logical, bite-sized chunks (e.g., by paragraph or slide).
* **Vector Database:** Embed these chunks using an embedding model and store them in a Vector DB (like Chroma, Pinecone, or Qdrant). Keep exercises and exams tagged separately from lecture materials so the LLM knows what to use for teaching vs. testing.
* **Summarization Engine:** Run a map-reduce summarization over the materials to create a "Global Context Document."

## Phase 2: High-Level Planning (The "Syllabus Agent")
Instead of reading all the raw content, the LLM uses the Global Context Document to plan.

1.  **Analyze Goals:** The LLM reviews the summaries of the lectures and the topics covered in previous exams to determine what the student *actually* needs to know.
2.  **Generate Learning Plan (JSON/YAML):** The LLM outputs a structured syllabus.
    * *Example Output:* Module 1 -> Lesson 1.1 -> Key Concepts -> Required Exercises -> Animation Concept.

## Phase 3: Detail Generation (The "Content Agent")
Now, the system loops through the generated learning plan. For *each individual lesson*, it executes the following steps:

1.  **Targeted Retrieval (RAG):** For "Lesson 1.1", query the Vector DB to pull the specific, highly relevant lecture chunks and previous exam questions related to that topic.
2.  **Draft Lesson Text:** The LLM writes the textbook-style explanation using only the retrieved context.
3.  **Generate Exercises:** Using the retrieved past exams as few-shot examples, the LLM generates new practice questions that mimic the difficulty and style of the real exams.

## Phase 4: Media Generation (The "Manim Agent Workflow")
### Sources
https://docs.manim.community/en/stable/index.html

This is the most complex part. LLMs often hallucinate Manim syntax because the library updates frequently. You need a **self-correcting execution loop**.

1.  **Prompting:** Pass the lesson text and a specific "Animation Concept" to the LLM, heavily prompting it with up-to-date Manim code examples (Few-Shot Prompting).
2.  **Code Generation:** The LLM outputs a Python script containing the Manim `Scene` class.
3.  **Secure Sandbox Execution:** Run the generated Python code in an isolated environment (like a Docker container) to render the MP4.
4.  **Feedback Loop (Crucial):** * *If the code errors out:* Capture the Python traceback/error message, send it back to the LLM, and ask it to fix the code. Set a max retry limit (e.g., 3 tries).
    * *If successful:* Save the MP4 file path to your database.

## Phase 5: Storage and Delivery
* **Database:** A relational database (PostgreSQL/SQLite) to store the structured course: Modules -> Lessons -> Text, Exercises, and Video URLs.
* **Frontend:** A simple web interface (React, Streamlit, or Next.js) where the user can log in, view the syllabus, read the lessons, attempt the exercises, and watch the Manim videos.

---

### Key Technical Stack Recommendations

* **Orchestration:** LangChain or LlamaIndex (for RAG and vector retrieval), or LangGraph/AutoGen (for managing the self-correcting Manim coding loop).
* **LLM Selection:** Use a high-tier reasoning model (like Gemini 1.5 Pro or GPT-4o) for the coding and planning phases. You can use smaller, faster models for summarization.
* **Animation:** Manim Community Edition (`manimce`).
* **Execution Environment:** Docker (absolutely necessary for running LLM-generated Python code safely).
