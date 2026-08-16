# TCS Prime Interview: Comprehensive Project Questions & Deep-Dive Guide

Candidate: **Sourajit Majumder**  
Target Role: **TCS Prime (AI/ML & Advanced Software Engineering)**  
Focus: Architectural Rigor, Engineering Justifications, Trade-offs, and Production Readiness.

---

## 📌 Table of Contents
1. [Project 1: YouTube Semantic Intelligence & Search Engine](#project-1-youtube-semantic-intelligence--search-engine)
2. [Project 2: Glaucoma Detection System — Deep Learning & Computer Vision](#project-2-glaucoma-detection-system--deep-learning--computer-vision)
3. [Project 3: AI-Powered Resume Screener & Job Matcher](#project-3-ai-powered-resume-screener--job-matcher)
4. [Hackathon & Full-Stack Projects (SIH 2025 & 100xDevs)](#hackathon--full-stack-projects)
5. [TCS Prime Behavioral & Project Defense Scenarios](#tcs-prime-behavioral--project-defense-scenarios)

---

# Project 1: YouTube Semantic Intelligence & Search Engine

> **Resume Bullet Points:**
> - *Built an automated pipeline for ingestion, chunking, and indexing of YouTube video transcripts, to support a grounded Q&A engine that answers with exact source citations.*
> - *Implemented a hybrid retrieval system (BM25 + semantic search with RRF reranking), reducing irrelevant/incorrect retrievals by approximately 40% versus keyword-only search.*
> - *Tech Stack: Python, FastAPI, ChromaDB, LiteLLM, SQLite, React*

---

### 1. Project Overview & Elevator Pitch
- **30-Second Pitch:**  
  *"I built a multimodal transcript intelligence engine that lets users ask complex questions across long YouTube videos and receives factual, grounded answers with exact timestamp citations. To solve the classic RAG retrieval failure where dense search misses exact keywords or technical terms, I engineered a Hybrid Retrieval pipeline combining sparse BM25 and dense vector search with Reciprocal Rank Fusion (RRF), improving retrieval relevance by 40%."*
- **2-Minute Pitch:**  
  *Walk through: Video URL input -> `youtube-transcript-api` extraction -> Semantic chunking with timestamp metadata -> ChromaDB indexing + BM25 index generation -> Query processing -> Hybrid search & RRF reranking -> Context injection to LiteLLM -> Streaming response to React frontend with clickable timestamp player.*

---

### 2. System Architecture & High-Level Design
```
[User Query (React UI)] 
          │
          ▼
[FastAPI REST API Server]
   ├── 1. Query Preprocessing / Embedding Generation
   ├── 2. Parallel Retrieval:
   │      ├── Sparse Search (BM25 Index over raw transcripts)
   │      └── Dense Vector Search (ChromaDB - Cosine Similarity)
   ├── 3. RRF (Reciprocal Rank Fusion) Reranker (k=60)
   ├── 4. Top-K Context Window Assembly (with Timestamps & Video IDs)
   ├── 5. LLM Prompt Construction (LiteLLM Proxy -> OpenAI / Anthropic / Groq)
   └── 6. Grounded Answer + Source Citations Streamed via SSE / JSON
```

#### Key Architectural Questions Interviewers Will Ask:
1. **Can you draw or explain the end-to-end architecture of your YouTube search system?**
   - *Expected Answer:* Explain the ingestion phase (fetching subtitles, cleaning timestamps, chunking) vs. the inference phase (query embedding, hybrid retrieval, fusion, synthesis).
2. **How do you preserve video timestamps across chunked text?**
   - *Expected Answer:* Each chunk is not just raw text; it is stored as a metadata object containing `video_id`, `start_time`, `end_time`, `chunk_index`, and `transcript_text`. When text is combined into 500-token chunks, the start time of the first line and end time of the last line form the chunk's temporal boundary.
3. **What chunking strategy did you use and why?**
   - *Options discussed:* Fixed-size (character/token count) vs. Recursive Character vs. Semantic/Sentence boundary vs. Video pause/segmentation.
   - *Defense:* Recursive Character Splitting with 500 tokens and 50 tokens overlap. 500 tokens preserves enough semantic context for the LLM while keeping embeddings dense and focused. Overlap prevents splitting critical thoughts across chunk borders.

---

### 3. Database Design & Storage Strategy
1. **Why ChromaDB AND SQLite? Why two databases?**
   - *Defense:* **Separation of Concerns**. ChromaDB is an in-memory/embedded vector store optimized for high-dimensional vector similarity search (HNSW index). SQLite is an embedded relational store used for metadata management, video ingestion status, user search history, and cached video transcript metadata.
2. **What does your SQLite schema look like?**
   - `videos`: `id (VARCHAR PK)`, `title`, `channel`, `duration`, `ingested_at`, `status`
   - `transcript_chunks`: `chunk_id (UUID PK)`, `video_id (FK)`, `start_timestamp (FLOAT)`, `end_timestamp (FLOAT)`, `text (TEXT)`
   - `search_logs`: `query_id`, `query_text`, `timestamp`, `latency_ms`
3. **How does ChromaDB index and query vectors?**
   - Uses **HNSW (Hierarchical Navigable Small World)** graphs with Cosine similarity or L2 distance. Space complexity is $O(N \cdot D)$, query time complexity is $O(\log N)$.

---

### 4. API & Backend Logic (FastAPI)
1. **Why FastAPI over Flask or Django?**
   - *Defense:* Native **asynchronous support (ASGI)** via `asyncio`/`uvicorn` for high-concurrency I/O-bound LLM API calls; automatic **Pydantic** request/response validation and OpenAPI Swagger documentation out of the box; significantly lower latency and higher requests-per-second (RPS) than Flask (WSGI).
2. **How did you handle long-running transcript ingestion without blocking the API?**
   - *Defense:* Used FastAPI `BackgroundTasks` or Celery/asynchronous worker pattern. The API returns an immediate `202 Accepted` with a `task_id`, and the ingestion pipeline runs in the background. The client polls the `/status/{task_id}` endpoint or listens via WebSocket/SSE.
3. **Explain how you implemented Streaming Responses for the LLM output.**
   - *Defense:* Used FastAPI's `StreamingResponse` with `text/event-stream` (Server-Sent Events) yielding chunks from LiteLLM's streaming generator to minimize Time-to-First-Token (TTFT).

---

### 5. Technology Choices & Justifications
1. **Why LiteLLM instead of raw OpenAI or LangChain?**
   - *Defense:* LiteLLM provides a unified OpenAI-compatible interface across 100+ LLMs (Groq, Anthropic, Bedrock, OpenAI) with built-in retry logic, fallback models, cost tracking, and zero framework lock-in. Unlike heavy LangChain abstractions, LiteLLM is lightweight and fast.
2. **Why Hybrid Search (BM25 + Dense) instead of Dense-only Vector Search?**
   - *Defense:* Vector embeddings excel at conceptual meaning ("how to repair a flat tire") but fail miserably at exact keyword queries, code snippets, acronyms, and product models (e.g., "RTX 4090", "CVE-2024-1234", or specific variable names). BM25 handles exact terms, while dense embeddings handle semantic meaning.
3. **What is RRF (Reciprocal Rank Fusion) and why is it better than simple score normalization?**
   - *Defense:* BM25 scores are unbounded $[0, \infty)$, while Cosine similarity is $[-1, 1]$. Normalizing both to $[0, 1]$ is unstable because score distributions vary per query. RRF bypasses score calibration by using positional ranks:
     $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
     where $k \approx 60$. It is robust, parameter-free, and prevents an outlier score from dominating the ranking.

---

### 6. Security Considerations
1. **How do you prevent Prompt Injection attacks?**
   - *Defense:* System prompt isolation with clear delimitation (`### CONTEXT: {context} ### USER QUERY: {query}`); strict instruction to answer *only* based on the provided context; output guardrails checking for system prompt leakage.
2. **How are API keys and sensitive environment variables managed?**
   - *Defense:* Loaded via `.env` using `pydantic-settings` / `python-dotenv`, never committed to version control, injected as environment variables in Docker.
3. **What happens if a user submits a malicious video URL or massive playlist?**
   - *Defense:* Strict URL regex validation; limiting ingestion to videos under a specific duration (e.g., max 2 hours); rate limiting per IP using Redis / token bucket algorithm.

---

### 7. Performance & Latency Optimization
1. **How did you achieve low latency in transcript retrieval?**
   - Embeddings generated in batches; ChromaDB HNSW indexing; query caching with Redis/LRU cache for identical queries.
2. **How did you quantify the "40% reduction in irrelevant/incorrect retrievals"?**
   - *Defense:* Created an evaluation dataset of 50 diverse queries (including exact keywords, acronyms, and conceptual queries). Measured **MRR@5 (Mean Reciprocal Rank)** and **Hit Rate@5**. Keyword search achieved 52% hit rate; hybrid search with RRF achieved 92% hit rate, representing a ~40% relative reduction in retrieval failures.

---

### 8. Scalability & Future Improvements
1. **If 100,000 videos need to be indexed, how does the architecture change?**
   - *Defense:* Migrate from embedded ChromaDB/SQLite to a distributed vector database (Milvus / Qdrant / Pinecone / pgvector) and PostgreSQL; use Apache Kafka / RabbitMQ for distributed transcript extraction workers; use asynchronous vector ingestion with Kubernetes auto-scaling.
2. **What future improvements would you make?**
   - Audio transcription via Whisper for videos without native transcripts; ColPali / multi-modal visual indexing for video slide frames; GraphRAG for cross-video thematic knowledge graph generation.

---
---

# Project 2: Glaucoma Detection System — Deep Learning & Computer Vision

> **Resume Bullet Points:**
> - *Built AI based Glaucoma detection system using ResNet-18 and U-Net with an AUC of 94.5% and accuracy of 87.1% which outperforms the strongest classical baseline (SVM, 78.9% AUC) by 15.6 percentage points.*
> - *Designed end-to-end ML pipeline comprising image pre-processing, data augmentation, model training, Grad-CAM explainability, and deployment as a real-time inference API.*
> - *Tech Stack: Python, PyTorch, ResNet-18, U-Net, Grad-CAM, scikit-learn, OpenCV*

---

### 1. Project Overview & Elevator Pitch
- **30-Second Pitch:**  
  *"I developed a medical-grade computer vision pipeline for automated Glaucoma screening from fundus eye images. By combining U-Net for optic disc and cup segmentation with a fine-tuned ResNet-18 for classification and Grad-CAM for clinical explainability, the model achieved a 94.5% ROC-AUC and 87.1% accuracy, beating a classical SVM baseline by 15.6 percentage points."*
- **2-Minute Pitch:**  
  *Explain medical motivation (glaucoma is an irreversible silent blinding disease caused by optic nerve damage), Cup-to-Disc Ratio (CDR) clinical metric, preprocessing (CLAHE contrast enhancement, ROI cropping), deep learning architecture (ResNet-18 + U-Net), explainability with heatmaps, and deployment.*

---

### 2. System Architecture & High-Level Design
```
[Raw Fundus Image (JPG/PNG)]
          │
          ▼
[Preprocessing Pipeline (OpenCV)]
   ├── 1. Green Channel Extraction & CLAHE (Contrast Limited Adaptive Histogram Equalization)
   ├── 2. Optic Disc Region of Interest (ROI) Localization & Cropping
   └── 3. Resizing (224x224 / 512x512) + Normalization (ImageNet stats)
          │
   ┌──────┴─────────────────────────────────┐
   ▼                                        ▼
[U-Net Segmentation Model]           [ResNet-18 Classification Model]
   ├── Optic Disc Mask                  ├── Feature Extraction (Conv layers + Residuals)
   ├── Optic Cup Mask                   ├── Binary Classification Head (Glaucoma / Normal)
   └── CDR Calculation (Vertical CDR)   └── Output Probability (Sigmoid)
   └──────────────┬─────────────────────────┘
                  ▼
[Explainability & Clinical Verification (Grad-CAM)]
   └── Activation Heatmap overlay on Optic Nerve Head
                  │
                  ▼
[FastAPI Real-Time Inference Endpoint]
   └── JSON Output: {prediction, probability, cdr_value, gradcam_heatmap_url}
```

---

### 3. Deep Technical & Machine Learning Questions
1. **Why is the Cup-to-Disc Ratio (CDR) clinically significant in Glaucoma?**
   - *Expected Answer:* Glaucoma damages retinal ganglion cells, leading to progressive optic nerve cupping. A vertical CDR $> 0.6$ or an asymmetry between eyes $> 0.2$ strongly indicates glaucomatous neuropathy.
2. **Why use ResNet-18 instead of a larger network like ResNet-50 or ViT?**
   - *Defense:* Medical fundus datasets (e.g., RIM-ONE, DRISHTI-GS, REFUGE) are relatively small (a few hundred to thousand images). Larger models like ResNet-50 or Vision Transformers have millions more parameters and severely overfit without massive data. ResNet-18 provides the ideal capacity-to-sample ratio, prevents overfitting, and offers sub-20ms inference time on CPU.
3. **What is the difference between ResNet-18 and U-Net in your pipeline?**
   - *Defense:* **ResNet-18** is a classification network (input: image, output: probability of disease). **U-Net** is a pixel-level semantic segmentation network (input: image, output: binary masks for optic cup and optic disc). U-Net computes the physical CDR, while ResNet-18 learns holistic visual patterns.
4. **Why did SVM with handcrafted features achieve only 78.9% AUC? What features did the SVM use?**
   - *Defense:* The SVM baseline utilized handcrafted texture and color features: Haralick GLCM (Gray-Level Co-occurrence Matrix) features, color histograms, and Gabor filters. Handcrafted features fail to capture subtle morphological neuroretinal rim thinning and vascular changes that CNN residual layers learn automatically.

---

### 4. Image Preprocessing & Computer Vision (OpenCV)
1. **Why extract the Green Channel from retinal fundus images?**
   - *Defense:* In fundus photography, the red channel is often over-saturated, and the blue channel has low signal-to-noise ratio due to lens absorption. The **green channel provides the highest contrast** between blood vessels, the optic disc, optic cup, and retinal background.
2. **What is CLAHE and why not standard Global Histogram Equalization?**
   - *Defense:* Standard Histogram Equalization operates globally and amplifies background sensor noise while blowing out bright regions like the optic disc. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** divides the image into small tiles (e.g., 8x8), equalizes contrast locally, and clips contrast exceeding a threshold to eliminate noise artifacts.
3. **What Data Augmentations did you apply and why?**
   - *Defense:* Random Horizontal/Vertical flips, random rotation ($\pm 15^\circ$), mild color jitter (brightness/contrast $\pm 0.1$), Affine transformations. Did *not* use extreme scaling or distortion because anatomic optic nerve proportions must remain intact.

---

### 5. Explainability with Grad-CAM
1. **How does Grad-CAM work mathematically?**
   - *Defense:* Grad-CAM computes the gradient of the target class score $y^c$ with respect to feature activation maps $A^k$ of the last convolutional layer:
     $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
     Then computes a weighted sum passed through ReLU:
     $$L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
     ReLU ensures only features that positively contribute to the target class are highlighted.
2. **Why is explainability critical in healthcare AI?**
   - *Defense:* Prevents "Clever Hans" effects where the model predicts glaucoma based on hospital watermarks, camera border artifacts, or patient age artifacts instead of the optic nerve head. Grad-CAM builds clinical trust by allowing ophthalmologists to visually verify the model's focus area.

---

### 6. Performance Metrics & Evaluation
1. **Why is ROC-AUC (94.5%) more informative than Accuracy (87.1%) in medical diagnosis?**
   - *Defense:* Medical datasets suffer from heavy **class imbalance** (far fewer disease cases than healthy). High accuracy can be achieved by predicting "healthy" for all cases. ROC-AUC evaluates the model across all classification thresholds, measuring the trade-off between Sensitivity (True Positive Rate / Recall) and Specificity ($1 - \text{False Positive Rate}$). In medical screening, high Sensitivity is critical to avoid missing positive patients.
2. **What loss functions did you use for U-Net and ResNet?**
   - *Defense:* For ResNet-18: **Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss)** or Focal Loss to handle class imbalance. For U-Net: **Dice Loss + BCE Loss (Combined Loss)** to optimize spatial overlap of small segmented regions (optic cup).

---
---

# Project 3: AI-Powered Resume Screener & Job Matcher

> **Resume Bullet Points:**
> - *Built an automated recruitment pipeline that parses job descriptions and candidate resumes (PDF/DOCX) into structured, schema-validated data using LLM-based extraction with Pydantic.*
> - *Built a candidate-ranking engine that scores resumes against job requirements, computing an overall match percentage, matching/missing skills, and experience-fit, producing a sorted shortlist of top candidates.*
> - *Tech Stack: Python, Groq API (Llama-3.3-70B), Pydantic, PyPDF, python-docx*

---

### 1. Project Overview & Elevator Pitch
- **30-Second Pitch:**  
  *"I developed an intelligent talent screening pipeline that replaces error-prone regex parsers with LLM-driven structured information extraction. Using Llama-3.3-70B via Groq's low-latency inference engine and Pydantic schema validation, the system converts unstructured PDF/DOCX resumes and JDs into structured candidate profiles and computes a multi-factor weighted match score to rank applicants."*

---

### 2. System Architecture & Workflow
```
[Resume (PDF/DOCX)] + [Job Description (Text/PDF)]
               │
               ▼
[Document Parsing Layer (PyPDF / python-docx)]
   ├── Text Extraction & Sanitization
   └── Unicode & Layout Normalization
               │
               ▼
[Structured LLM Extraction Engine]
   ├── Prompt with Strict JSON Schema
   ├── Groq API (Llama-3.3-70B - Ultra Fast Inference)
   └── Pydantic Schema Validation & Type Coercion
               │
               ▼
[Structured Candidate & JD Profile]
   ├── Skills List (Normalized to Taxonomy)
   ├── Years of Experience (Calculated from Dates)
   ├── Education Degree & Field of Study
   └── Key Projects & Domain Alignment
               │
               ▼
[Candidate Ranking Engine]
   ├── 1. Hard Skill Match Score (Jaccard / Weighted Keyword Vector)
   ├── 2. Experience Fit Score (Linear Penalty Function for shortfall)
   ├── 3. Semantic Relevance Score (Cosine similarity of project summaries)
   └── Composite Score = $w_1 \cdot S_{skill} + w_2 \cdot S_{exp} + w_3 \cdot S_{semantic}$
               │
               ▼
[Ranked Shortlist Dashboard & Missing Skills Gap Report]
```

---

### 3. Deep Technical & Engineering Questions
1. **Why use an LLM for resume parsing instead of traditional regex/Spacy NER?**
   - *Defense:* Resumes have infinite layout variations (multi-column tables, varied date formats, non-standard section headers, creative phrasing). Traditional rule-based/regex parsers break on non-standard layouts. LLMs understand semantic context (e.g., recognizing that "Built microservices using Spring Boot" implies Java proficiency even if "Java" isn't explicitly in a skills bullet point).
2. **Why use Pydantic for schema validation? What happens when the LLM outputs malformed JSON?**
   - *Defense:* LLMs are non-deterministic and occasionally output markdown blocks, extra commentary, or mismatched types. Pydantic enforces strict type validation (e.g., `years_of_experience: float`, `skills: List[str]`). If validation fails, the error is caught, and an automated retry mechanism reprompts the LLM with the exact validation error message or forces `response_format={"type": "json_object"}`.
3. **Why did you choose Groq (Llama-3.3-70B) over OpenAI GPT-4o?**
   - *Defense:* **Inference Speed & Cost Efficiency**. Groq's LPU (Language Processing Unit) delivers 300+ tokens/second inference speed with near-zero latency compared to 30-50 tokens/sec on cloud GPUs. Llama-3.3-70B matches GPT-4-class reasoning on extraction tasks at a fraction of the cost, making batch resume processing viable.
4. **How do you calculate the match score? Explain your mathematical formula.**
   - *Defense:* 
     $$\text{Final Score} = 0.45 \cdot S_{\text{skills}} + 0.30 \cdot S_{\text{experience}} + 0.25 \cdot S_{\text{semantic\_fit}}$$
     - $S_{\text{skills}} = \frac{|\text{Candidate Skills} \cap \text{Required Skills}|}{|\text{Required Skills}|}$ (weighted by core vs nice-to-have).
     - $S_{\text{experience}} = \min\left(1.0, \frac{\text{Candidate Exp (yrs)}}{\text{Required Exp (yrs)}}\right)$.
     - $S_{\text{semantic\_fit}}$ compares project domain vectors with JD requirements.

---

### 4. Edge Cases & Challenges
1. **How do you handle multi-column PDF resumes where standard extractors read text across columns horizontally?**
   - *Defense:* PyPDF can sometimes mix column text. Implemented bounding-box block extraction using layout-aware parsing or structured chunking before feeding text to the LLM.
2. **How do you prevent Algorithmic Bias in resume screening?**
   - *Defense:* **Anonymization & Blind Screening**. The extraction prompt strips PII (Personally Identifiable Information) including Candidate Name, Gender, Age, Photograph, Ethnicity, and Postal Address. The ranking engine evaluates strictly on verifiable technical skills, project impact, and domain experience.

---
---

# Hackathon & Full-Stack Projects

### Smart India Hackathon (SIH) 2025 Prototype
- **Interview Question:** *"Tell me about your SIH 2025 Hackathon experience. What problem did you solve, and what was your role?"*
- **Defense Strategy:**
  - State the problem statement (e.g., AI-based grievance classification, automated civic anomaly detection, or intelligent resource scheduling).
  - Highlight teamwork under pressure: rapid prototyping in 36 hours, Git branching workflow, handling API rate limits, dividing frontend, backend, and AI pipeline responsibilities.
  - Emphasize your individual contribution: built the FastAPI backend and AI model integration.

### Full-Stack (100xDevs) — React, Node.js, Express, PostgreSQL
- **Interview Question:** *"You have Node.js/PostgreSQL from 100xDevs and FastAPI/MongoDB in your skills. Compare when you would choose Node.js + PostgreSQL vs. Python + FastAPI + MongoDB."*
- **Defense Strategy:**
  - *Node.js + PostgreSQL:* Ideal for relational data with complex ACID transaction requirements (financial ledger, e-commerce cart/order system) and I/O-intensive real-time event-driven apps (chat apps via WebSockets).
  - *FastAPI + MongoDB:* Ideal for AI/ML pipelines where native Python tensor libraries (PyTorch, NumPy) are required, with unstructured/semi-structured dynamic JSON data schemas.

---
---

# TCS Prime Behavioral & Project Defense Scenarios

1. **"What was the single most difficult bug you encountered across your projects, and how did you resolve it?"**
   - *Framework (STAR):*
     - **Situation:** In the Glaucoma project, the model initially achieved 92% validation accuracy but failed on external clinical test images.
     - **Task:** Diagnose the root cause of domain shift and poor generalization.
     - **Action:** Ran Grad-CAM visualization and discovered the model was focusing on the dark circular borders of the fundus camera aperture rather than the optic cup. Applied automated circular mask cropping to extract only the retinal ROI and normalized lighting with CLAHE.
     - **Result:** Generalization AUC rose to 94.5%, and Grad-CAM confirmed focus on the optic disc.
2. **"If your project had to be deployed to 1 million daily active users tomorrow, what would break first?"**
   - *Answer:* The synchronous in-memory vector search and single-instance FastAPI server. ChromaDB would run out of RAM, and API response times would spike. Solution: Move to distributed vector storage (Qdrant/Milvus with disk-backed quantization), place FastAPI behind an NGINX load balancer with Gunicorn Uvicorn workers in Docker containers on Kubernetes, and introduce Redis caching for frequent queries.
