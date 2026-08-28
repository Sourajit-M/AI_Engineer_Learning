# 🧠 TCS Prime Interview: 100+ Deep-Dive Follow-Up Questions & Answers

This master guide prepares you for the deep, chain-of-thought follow-up questions TCS Prime interviewers ask when you mention any technology, framework, database, or concept from your resume and project (`05_resume_parser.py`).

---

## 📑 Table of Categories
1. [Groq, LLMs & GenAI Concepts](#1-groq-llms--genai-concepts)
2. [Pydantic V2 & Data Modeling](#2-pydantic-v2--data-modeling)
3. [Python Core & Advanced Language Internals](#3-python-core--advanced-language-internals)
4. [Document Parsing & Ingestion (PyPDF & python-docx)](#4-document-parsing--ingestion-pypdf--python-docx)
5. [Databases: SQL vs NoSQL vs Vector Databases](#5-databases-sql-vs-nosql-vs-vector-databases)
6. [System Design, Scalability & Architecture](#6-system-design-scalability--architecture)
7. [Core CS: Data Structures & Algorithms (DSA)](#7-core-cs-data-structures--algorithms-dsa)
8. [Core CS: Object-Oriented Programming (OOPs)](#8-core-cs-object-oriented-programming-oops)
9. [Core CS: Operating Systems (OS)](#9-core-cs-operating-systems-os)
10. [Core CS: Computer Networks (CN) & Web Technologies](#10-core-cs-computer-networks-cn--web-technologies)
11. [Cross-Project & Resume Deep Dives](#11-cross-project--resume-deep-dives)
12. [Managerial & Situational Follow-Ups](#12-managerial--situational-follow-ups)

---

## 1. Groq, LLMs & GenAI Concepts

### Follow-up on: *"I used Groq and Llama-3.3-70B."*

#### Q1. What is Groq? How does an LPU differ from a GPU and CPU?
> **Answer:**
> - **CPU (Central Processing Unit):** Optimized for low-latency sequential processing with large caches and complex branch prediction, but low parallel throughput.
> - **GPU (Graphics Processing Unit):** Massive parallel SIMD (Single Instruction, Multiple Data) compute with High Bandwidth Memory (HBM). Great for training, but inference suffers from memory bandwidth bottlenecks (von Neumann bottleneck).
> - **Groq LPU (Language Processing Unit):** A **Tensor Streaming Processor (TSP)** designed specifically for sequential inference. It features deterministic execution without traditional caches or branch predictors, utilizing ultra-fast on-chip SRAM (230 TB/s bandwidth). This allows it to achieve **300–500 tokens/sec** for LLMs, virtually eliminating token-generation latency.

#### Q2. What is Llama-3.3-70B? Explain its underlying architecture.
> **Answer:**
> "Llama-3.3-70B is an open-weights autoregressive Large Language Model developed by Meta based on the **Transformer Decoder-Only Architecture**.
> Key architectural pillars:
> 1. **Grouped Query Attention (GQA):** Uses 8 Key-Value heads for 64 Query heads, drastically reducing memory bandwidth during autoregressive decoding and speeding up inference.
> 2. **Rotary Positional Embeddings (RoPE):** Encodes relative token positions by rotating query and key representations in the complex plane, supporting up to 128k context length.
> 3. **SwiGLU Activation Function:** Non-linear gating function that outperforms standard ReLU/GELU.
> 4. **128k Vocabulary (tiktoken-based BPE):** Improves tokenization efficiency across code and multilingual text."

#### Q3. What do hyperparameters like Temperature, Top-P, and Top-K do?
> **Answer:**
> - **Temperature ($T$):** Scales the logits before softmax: $P(w_i) = \frac{e^{z_i / T}}{\sum e^{z_j / T}}$. Lower temperature ($T \to 0$) makes the output deterministic and greedy (ideal for JSON extraction). Higher temperature ($T > 0.7$) flattens the distribution for creative output.
> - **Top-P (Nucleus Sampling):** Selects only the smallest set of tokens whose cumulative probability exceeds $P$ (e.g., $P=0.9$). Dynamically expands or narrows candidate token pool.
> - **Top-K:** Restricts candidate selection to the fixed top $K$ most probable tokens.

#### Q4. What is Hallucination in LLMs? How did you mitigate it in this project?
> **Answer:**
> "Hallucination is when an LLM generates factually incorrect or ungrounded assertions with high confidence.
> In my project, I mitigated it by:
> 1. **Strict Grounding Prompts:** Explicit negative constraints: *'Do not invent information. If missing, return null or empty list.'*
> 2. **Pydantic Schema Constrained Generation:** Enforced structured JSON output via `response_format={"type": "json_object"}`.
> 3. **Source Context Injection:** Passing only extracted raw resume text into the prompt context rather than asking open-ended questions."

#### Q5. How does `response_format={"type": "json_object"}` work under the hood?
> **Answer:**
> "It activates **Grammar-Constrained Decoding** on the inference engine. During the token sampling step, the engine masks out any tokens from the vocabulary that would produce invalid JSON syntax (e.g., forbidding unquoted keys, trailing commas, or non-matching braces), guaranteeing syntactically valid JSON."

---

## 2. Pydantic V2 & Data Modeling

### Follow-up on: *"I used Pydantic for data validation."*

#### Q6. What is Pydantic and why use it over standard Python dictionaries or dataclasses?
> **Answer:**
> - **Standard Dictionaries:** Untyped, error-prone, no validation, rely on fragile string keys (`data['skills']` throws `KeyError`).
> - **Python `dataclasses`:** Provide type hints at write-time, but **do not perform runtime type coercion or validation**. Assigning `"10"` to an `int` field is silently accepted.
> - **Pydantic:** Enforces strict runtime data validation, automatic type casting/coercion, nested object serialization/deserialization, and automatic JSON Schema generation."

#### Q7. What happens when you do `job_desc = JobDesc(**job_data)`?
> **Answer:**
> 1. Pydantic unpacks the dictionary and matches keys to model fields.
> 2. It runs validation logic compiled in **Rust (`pydantic-core`)**.
> 3. It performs **type coercion** (e.g., parsing `"3.0"` string to `3.0` float if possible).
> 4. If any field violates constraints (e.g., missing required field or wrong type), it raises a `ValidationError` containing the exact error location.
> 5. If valid, it returns an immutable or mutable typed model instance."

#### Q8. What is the difference between Pydantic V1 and Pydantic V2?
> **Answer:**
> - **Core Engine:** Pydantic V2 rewritten with a native **Rust core (`pydantic-core`)**, achieving **5x to 50x faster validation**.
> - **Method Renaming:** `.dict()` became `.model_dump()`, `.json()` became `.model_dump_json()`, and `.schema()` became `.model_json_schema()`.
> - **Validators:** `@validator` replaced with modern `@field_validator` and `@model_validator`."

#### Q9. How do you define optional fields or default values in Pydantic?
> **Answer:**
> ```python
> from pydantic import BaseModel, Field
> 
> class Example(BaseModel):
>     name: str                          # Required
>     age: int | None = None             # Optional (defaults to None)
>     skills: list[str] = Field(default_factory=list) # Default empty list
>     score: float = Field(ge=0, le=100) # Validated range: 0 <= score <= 100
> ```

---

## 3. Python Core & Advanced Language Internals

### Follow-up on: *"My primary language for this project was Python."*

#### Q10. How does Python manage memory? Explain Garbage Collection.
> **Answer:**
> "Python uses a two-fold memory management strategy:
> 1. **Reference Counting:** Every Python object maintains an internal reference counter (`ob_refcnt`). When an object is referenced, its count increases; when dereferenced, it decreases. When `refcount == 0`, the memory is immediately deallocated.
> 2. **Cyclic Garbage Collector (`gc` module):** Reference counting fails on reference cycles (Object A references B, and B references A). Python's cyclic GC periodically runs across 3 generations (Gen 0, Gen 1, Gen 2) using a doubly linked list to detect and collect unreachable cyclic references."

#### Q11. What is the Global Interpreter Lock (GIL)? How does it affect concurrency?
> **Answer:**
> "The **GIL** is a mutex that allows only one native thread to execute Python bytecode at a time in CPython.
> - **Impact on CPU-bound tasks:** Multi-threading does not provide speedup because threads contend for the single GIL. Use `multiprocessing` or native C/Rust extensions instead.
> - **Impact on I/O-bound tasks:** The GIL is released during I/O operations (network requests, file reading, database queries), so `threading` or `asyncio` achieves high concurrency."

#### Q12. What is the difference between `asyncio`, `threading`, and `multiprocessing`?
> **Answer:**
> | Feature | `asyncio` | `threading` | `multiprocessing` |
> | :--- | :--- | :--- | :--- |
> | **Model** | Single-threaded event loop (Cooperative multitasking) | Multi-threading managed by OS (Preemptive) | Separate OS processes with separate memory spaces |
> | **GIL Affected?** | No (Single thread) | Yes (Only 1 thread runs at a time) | No (Each process has its own GIL & memory) |
> | **Best For** | High-concurrency network I/O (e.g., calling 500 LLM APIs) | Disk I/O & legacy blocking code | CPU-heavy workloads (e.g., Computer Vision, ResNet training) |

#### Q13. Explain how Timsort works (used in `results.sort()`).
> **Answer:**
> - **Timsort** is Python's standard sorting algorithm (invented by Tim Peters).
> - It is a hybrid stable sorting algorithm derived from **Merge Sort and Insertion Sort**.
> - It finds natural ordered sequences ('runs') in data. For small runs, it uses Insertion Sort; for larger collections, it merges runs using an optimized Merge Sort.
> - **Time Complexity:** Best case $O(N)$ (already sorted), Average & Worst case $O(N \log N)$. Space complexity: $O(N)$."

#### Q14. What are Generators and how do they differ from normal functions?
> **Answer:**
> - Normal functions return a single value and terminate their stack frame (`return`).
> - Generators yield values lazily on demand using the `yield` keyword, maintaining their state between calls.
> - **Memory Advantage:** Instead of loading a 10 GB file or 100,000 resumes into RAM at once, a generator yields one resume at a time with $O(1)$ memory."

---

## 4. Document Parsing & Ingestion (PyPDF & python-docx)

### Follow-up on: *"I parsed PDFs using PyPDF and Word docs using python-docx."*

#### Q15. How does a PDF store text, and why is PDF text extraction challenging?
> **Answer:**
> "PDF is a visual layout presentation format, not a semantic text format. It stores glyphs and characters at explicit 2D coordinate positions `(x, y)` on a canvas.
> **Challenges:**
> 1. **No native concept of paragraphs/words:** Text streams are arbitrary chunks; two-column layouts often get merged linearly (reading left-column line 1, then right-column line 1).
> 2. **Ligatures & Custom Encodings:** Characters like 'fi', 'fl' may be encoded as single non-standard font glyphs.
> 3. **Scanned Documents:** Contain only raster images without underlying text streams."

#### Q16. How does `python-docx` read `.docx` files under the hood?
> **Answer:**
> "A `.docx` file is actually a **ZIP archive** containing XML files following the OpenXML standard. The core document content resides in `word/document.xml`. `python-docx` parses this XML tree, traversing paragraph elements (`<w:p>`), text runs (`<w:r>`), and table rows/cells (`<w:tbl>`, `<w:tr>`, `<w:tc>`)."

---

## 5. Databases: SQL vs NoSQL vs Vector Databases

### Follow-up on: *"I would store candidate data in MongoDB / PostgreSQL."*

#### Q17. Why choose MongoDB over MySQL for storing parsed resumes?
> **Answer:**
> 1. **Schema Polymorphism:** Resumes vary wildly—some have 10 skills, some have 50; some list 4 internships with project links, others list none. MongoDB's JSON/BSON document model handles polymorphic data without expensive schema migrations.
> 2. **Nested Document Storage:** Nested Pydantic structures like `experiences: list[Experience]` map directly to embedded document arrays in MongoDB without needing multiple foreign-key join tables.
> 3. **Horizontal Scalability:** MongoDB natively supports sharding and horizontal scaling for high-volume unstructured data."

#### Q18. When would you choose PostgreSQL (SQL) over MongoDB (NoSQL)?
> **Answer:**
> - When the system requires **strict relational integrity and multi-table ACID transactions** (e.g., financial transactions, billing, user role-based access control).
> - When complex multi-table `JOIN` queries and analytical aggregations are standard.
> - When data schema is stable, highly structured, and normalized."

#### Q19. What is Database Indexing? How does a B-Tree index work?
> **Answer:**
> "An **index** is a data structure (commonly a self-balancing **B-Tree** or B+ Tree) that allows the database engine to find specific records in $O(\log N)$ time instead of performing an $O(N)$ full table scan.
> - In a B+ Tree, leaf nodes are linked in a sequential doubly linked list, enabling both rapid point lookups and efficient range scans."

#### Q20. What are ACID properties? Explain each with an example.
> **Answer:**
> - **Atomicity:** All operations in a transaction succeed or all roll back (All-or-Nothing). *Example: Deducting candidate interview credits and creating an interview slot must both succeed.*
> - **Consistency:** The database moves from one valid state to another, preserving all schema constraints.
> - **Isolation:** Concurrent transactions execute independently without interfering with each other (controlled via isolation levels: Read Committed, Repeatable Read, Serializable).
> - **Durability:** Once committed, transaction data is permanently written to non-volatile disk/WAL even if power fails."

#### Q21. What is a Vector Database (ChromaDB / Qdrant) and how does vector search work?
> **Answer:**
> "A Vector Database indexes high-dimensional dense floating-point arrays (embeddings) generated by ML models.
> - **Similarity Metrics:** Cosine Similarity, Dot Product, Euclidean ($L2$) Distance.
> - **Approximate Nearest Neighbor (ANN) Algorithms:** Exact search is $O(N \cdot D)$ (too slow). ANN algorithms like **HNSW (Hierarchical Navigable Small World)** construct multi-layer graphs to achieve sub-linear $O(\log N)$ nearest-neighbor retrieval."

---

## 6. System Design, Scalability & Architecture

### Follow-up on: *"How would you scale this to enterprise production?"*

#### Q22. Explain how to design a distributed asynchronous resume processing pipeline.
> **Answer:**
> 1. **Client / Gateway:** React Frontend + FastAPI API Gateway behind Nginx Load Balancer.
> 2. **Object Storage:** Resumes uploaded to AWS S3, generating an event notification.
> 3. **Message Queue:** S3 event pushes job ID to **RabbitMQ / Redis Queue**.
> 4. **Worker Pool:** Containerized **Celery Workers** pull jobs from the queue, execute extraction, call LLM APIs with retry/circuit breaker logic.
> 5. **Persistence & Search:** Parsed profiles stored in MongoDB; vector embeddings stored in Qdrant.
> 6. **Notification:** WebSockets or SSE push live match updates to the recruiter dashboard."

#### Q23. What is Rate Limiting? What algorithms exist?
> **Answer:**
> "Rate limiting controls the rate of traffic sent or received on a network/API to prevent abuse and adhere to third-party API quotas.
> **Algorithms:**
> 1. **Token Bucket:** Tokens added at fixed rate; requests consume tokens. Allows bursts up to bucket capacity.
> 2. **Leaky Bucket:** Requests enter a queue and leak out at a constant rate. Smooths out traffic spikes.
> 3. **Sliding Window Log:** Tracks timestamps of all requests; highly accurate but memory heavy.
> 4. **Sliding Window Counter:** Hybrid memory-efficient sliding approximation."

---

## 7. Core CS: Data Structures & Algorithms (DSA)

### Potential Pen & Paper Coding Questions in TCS Prime

#### Q24. Explain Binary Search. Write its iterative code and state its complexity.
> **Answer:**
> "Binary search operates on sorted arrays by dividing the search interval in half repeatedly."
> ```python
> def binary_search(arr: list[int], target: int) -> int:
>     left, right = 0, len(arr) - 1
>     while left <= right:
>         mid = left + (right - left) // 2  # Prevents integer overflow
>         if arr[mid] == target:
>             return mid
>         elif arr[mid] < target:
>             left = mid + 1
>         else:
>             right = mid - 1
>     return -1
> ```
> - **Time Complexity:** $O(\log N)$, **Space Complexity:** $O(1)$."

#### Q25. What is the difference between Array and LinkedList in memory?
> **Answer:**
> - **Array:** Stored in **contiguous memory blocks**. Allows $O(1)$ random access via indexing. Insertion/deletion in middle takes $O(N)$ due to element shifting. Fixed size in static languages.
> - **LinkedList:** Stored in **non-contiguous memory**. Each node contains data and a pointer to next node. $O(N)$ sequential access. Insertion/deletion is $O(1)$ once pointer position is known. Overhead of extra pointer memory."

#### Q26. Explain Dynamic Programming. What are Memoization and Tabulation?
> **Answer:**
> "DP solves complex problems by breaking them into overlapping subproblems and optimal substructures, storing intermediate results to avoid redundant calculations.
> - **Memoization (Top-Down):** Recursion with a cache (e.g., Python `@lru_cache` or hash table).
> - **Tabulation (Bottom-Up):** Iterative approach filling a DP table from base cases up to target."

---

## 8. Core CS: Object-Oriented Programming (OOPs)

### Follow-up on: *"Core CS: OOPs"*

#### Q27. Explain the 4 Pillars of OOPs with real-world examples.
> **Answer:**
> 1. **Encapsulation:** Bundling data and methods into a single unit (class) and restricting direct access using private/protected modifiers. *Example: Pydantic `BaseModel` hiding internal validation logic.*
> 2. **Abstraction:** Hiding implementation complexity and exposing only relevant interfaces. *Example: Calling `client.chat.completions.create()` without worrying about TCP sockets or GPU clusters.*
> 3. **Inheritance:** Mechanism where a child class acquires properties of a parent class. *Example: `class JobDesc(BaseModel)` inheriting serialization and validation methods from Pydantic `BaseModel`.*
> 4. **Polymorphism:** Ability of an entity to take multiple forms.
>    - *Compile-Time (Static):* Method Overloading (same name, different signature).
>    - *Run-Time (Dynamic):* Method Overriding (child class redefines parent method)."

#### Q28. What is the difference between an Abstract Class and an Interface?
> **Answer:**
> | Feature | Abstract Class | Interface |
> | :--- | :--- | :--- |
> | **Definition** | A class that cannot be instantiated and can contain both abstract and concrete methods | A complete contract containing only method signatures (in Java) or purely abstract methods |
> | **Multiple Inheritance** | Most languages (Java/C#) do not allow multiple class inheritance | A class can implement multiple interfaces |
> | **State/Variables** | Can have instance variables and state | In Java, fields are `public static final` constants by default |

---

## 9. Core CS: Operating Systems (OS)

### Follow-up on: *"Core CS: OS"*

#### Q29. What is the difference between a Process and a Thread?
> **Answer:**
> - **Process:** An independent program in execution with its own dedicated virtual address space, memory, file descriptors, and security context. Process switching has high overhead.
> - **Thread:** A lightweight unit of execution within a process that shares the parent process's memory space and open files, but has its own stack and registers. Context switching is fast."

#### Q30. What is a Deadlock? What are the 4 Coffman Conditions?
> **Answer:**
> "A **Deadlock** is a state where a set of processes are blocked because each process holds a resource and waits for another resource held by another process.
> **4 Necessary Coffman Conditions:**
> 1. **Mutual Exclusion:** Resources cannot be shared simultaneously.
> 2. **Hold and Wait:** A process holds at least one resource and is waiting to acquire others.
> 3. **No Preemption:** Resources cannot be forcibly taken; they must be released voluntarily.
> 4. **Circular Wait:** A closed chain of processes exists such that each holds a resource needed by the next."

#### Q31. What is Virtual Memory and Paging? What is Thrashing?
> **Answer:**
> - **Virtual Memory:** An OS abstraction creating the illusion of large, contiguous physical memory by combining RAM with disk storage (Swap Space).
> - **Paging:** Memory management scheme dividing virtual memory into fixed-size blocks called *Pages* and physical RAM into *Page Frames*.
> - **Thrashing:** A state where the OS spends more time swapping pages in and out of disk than executing actual instructions due to insufficient physical RAM."

---

## 10. Core CS: Computer Networks (CN) & Web Technologies

### Follow-up on: *"FastAPI, REST APIs, HTTP, Networks"*

#### Q32. Explain the difference between TCP and UDP.
> **Answer:**
> - **TCP (Transmission Control Protocol):** Connection-oriented, reliable, guarantees packet ordering, handles congestion and flow control via **3-Way Handshake (SYN -> SYN-ACK -> ACK)**. Used in HTTP/HTTPS, WebSockets, DB connections.
> - **UDP (User Datagram Protocol):** Connectionless, unreliable, low latency, no ordering or retransmissions. Used in live video streaming, DNS lookups, online gaming."

#### Q33. What happens when you type `https://api.groq.com` in your browser?
> **Answer:**
> 1. **DNS Lookup:** Resolves domain `api.groq.com` to an IP address (Browser cache $\to$ OS cache $\to$ Resolver $\to$ Root/TLD/Authoritative DNS).
> 2. **TCP 3-Way Handshake:** Establishes reliable connection on Port 443.
> 3. **TLS/SSL Handshake:** Authenticates server certificate and negotiates symmetric encryption keys.
> 4. **HTTP Request & Response:** Browser sends encrypted `GET` request; server returns HTTP response."

#### Q34. What are common HTTP Status Codes?
> **Answer:**
> - `200 OK`: Request succeeded.
> - `201 Created`: Resource successfully created (common on `POST`).
> - `400 Bad Request`: Client sent malformed payload.
> - `401 Unauthorized`: Missing or invalid authentication token.
> - `403 Forbidden`: Authenticated, but lacking permission.
> - `404 Not Found`: Resource does not exist.
> - `429 Too Many Requests`: Rate limit exceeded.
> - `500 Internal Server Error`: Unhandled server crash.
> - `502 Bad Gateway` / `503 Service Unavailable`: Upstream service down."

---

## 11. Cross-Project & Resume Deep Dives

### Follow-up on: *"YouTube Semantic Intelligence (RRF & BM25)"*

#### Q35. What is BM25 and why combine it with Semantic Search?
> **Answer:**
> - **BM25 (Best Matching 25):** Probabilistic sparse keyword search based on TF-IDF. Excellent for exact phrase, part number, and acronym matching.
> - **Semantic Search (Dense Embeddings):** Captures conceptual intent and synonyms, but can miss exact keyword lookups.
> - **Hybrid Search with RRF (Reciprocal Rank Fusion):** Combines rank positions from both retrievers using $RRF\_Score(d) = \sum \frac{1}{k + r_i(d)}$ to get the best of both worlds."

### Follow-up on: *"Glaucoma Detection System (ResNet-18 & Grad-CAM)"*

#### Q36. Why use ResNet-18 instead of a standard deep CNN? What is a Residual Connection?
> **Answer:**
> "In deep vanilla CNNs, adding more layers causes the **Vanishing/Exploding Gradient Problem** and degradation. ResNet introduces **Skip Connections (Residual Connections)**: $\mathcal{H}(x) = \mathcal{F}(x) + x$. This allows gradients to flow directly backwards through the identity shortcut, enabling effective training of deep architectures."

#### Q37. What is Grad-CAM?
> **Answer:**
> "**Gradient-weighted Class Activation Mapping (Grad-CAM)** provides visual explainability for CNNs by computing the gradients of the target class score with respect to the feature map activations of the final convolutional layer, producing a coarse 2D heatmap highlighting the exact regions of the image that influenced the model's prediction."

---

## 12. Managerial & Situational Follow-Ups

#### Q38. Why TCS, and specifically why the TCS Prime role?
> **Answer:**
> "TCS is a global leader driving digital transformation for Fortune 500 enterprises. The **TCS Prime role** is specifically tailored for engineers who demonstrate strong problem-solving, architectural acumen, and emerging tech capabilities in AI/ML and Cloud. 
> 
> With my hands-on experience in building end-to-end GenAI pipelines, deep learning computer vision models, and solid foundation in 450+ LeetCode problems, I am ready to contribute directly to high-impact enterprise AI solutions at TCS from Day 1."

#### Q39. Are you willing to relocate or work across diverse technology stacks?
> **Answer:**
> "Yes, absolutely. I view relocation as an opportunity to collaborate with diverse teams and expand my professional horizon. As demonstrated by my projects spanning AI/ML, backend APIs, and ServiceNow administration, I am language-agnostic and adapt quickly to any technology stack required by business goals."
