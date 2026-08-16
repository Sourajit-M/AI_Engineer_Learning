# TCS Prime Interview: Exhaustive Multi-Level Follow-Up Questions

Candidate: **Sourajit Majumder**  
Target Role: **TCS Prime (AI/ML & Advanced Software Engineering)**  
Pattern: **"If you say X $\rightarrow$ Be prepared for Y & Z"**

---

## 🧭 How Interviewers Trap Candidates
In a TCS Prime interview, interviewers will rarely stop at your surface answer. They use **depth-probing questioning chains**. If you use a keyword, they will peel back 3 to 4 layers until they hit fundamental computer science, math, or low-level implementation details.

---

# 1. Databases & Data Stores (ChromaDB, SQLite, MongoDB, PostgreSQL)

### 💬 "I used ChromaDB for vector storage in my RAG project."
- **Level 1:** Why ChromaDB? Why not Pinecone, Weaviate, Milvus, or FAISS?
  - *Answer:* ChromaDB is lightweight, open-source, and can run fully embedded in-process with Python (no extra cloud cost or network roundtrip). Pinecone is managed/cloud-only; Milvus is heavier distributed infrastructure; FAISS is purely an indexing library without native CRUD metadata filtering.
- **Level 2:** How does vector indexing actually work under the hood?
  - *Answer:* ChromaDB defaults to **HNSW (Hierarchical Navigable Small World)** graphs. It builds multi-layer graphs where top layers have long links (fast coarse search) and bottom layers have dense links (fine-grained nearest neighbor search). Query complexity is $O(\log N)$.
- **Level 3:** What distance metric did you use? Why Cosine Similarity over Euclidean (L2) or Dot Product?
  - *Answer:* Cosine similarity measures the angle between vectors, normalizing for document length ($A \cdot B / (\|A\| \|B\|)$). If embeddings are unit-normalized ($L_2 = 1$), Cosine Similarity and Dot Product are mathematically equivalent and Dot Product is faster to compute. L2 distance is sensitive to magnitude variations.
- **Level 4:** What are the limitations of vector search?
  - *Answer:* Vector search suffers from the "Curse of Dimensionality", cannot handle exact keyword matching (acronyms, code identifiers, phone numbers), and pure vector search is computationally expensive without quantization (IVF/PQ/HNSW).

---

### 💬 "I used SQLite for metadata and MongoDB in other backend projects."
- **Level 1:** Why use MongoDB (NoSQL) over MySQL or PostgreSQL (SQL)?
  - *Answer:* MongoDB provides schema flexibility for polymorphic, evolving document structures (nested JSON like user profiles or dynamic metadata). SQL is preferred when schema is rigid, relationships are complex, and multi-table ACID transactions are mandatory.
- **Level 2:** How does MongoDB store data internally?
  - *Answer:* MongoDB stores data in **BSON** (Binary JSON) format on disk using the **WiredTiger storage engine**, which organizes data using **B-Trees** and provides document-level concurrency and snappy compression.
- **Level 3:** What is Indexing? How does a B-Tree index work?
  - *Answer:* An index is an auxiliary data structure that reduces disk I/O from $O(N)$ full table scans to $O(\log N)$ tree traversals. In a B-Tree / B+ Tree, keys are sorted in balanced nodes, allowing efficient range queries and point lookups.
- **Level 4:** What is the difference between a Clustered Index and a Non-Clustered Index?
  - *Answer:* A **Clustered Index** determines the physical storage order of rows on disk (only one per table, usually the Primary Key). A **Non-Clustered Index** is stored separately with pointers (row IDs or primary keys) referencing the actual data rows.
- **Level 5:** Explain the CAP Theorem. Which two does MongoDB choose?
  - *Answer:* CAP states a distributed system can guarantee at most 2 out of 3: Consistency, Availability, Partition Tolerance. By default, MongoDB chooses **CP (Consistency & Partition Tolerance)**—during network partitions, writes to minority partitions are rejected until a primary is elected.
- **Level 6:** What are ACID properties? How does MongoDB support ACID?
  - *Answer:* **Atomicity, Consistency, Isolation, Durability**. Since MongoDB 4.0, multi-document ACID transactions are supported across replica sets using two-phase commit protocols.

---

# 2. Information Retrieval, RAG & LLMs

### 💬 "I implemented Hybrid Retrieval using BM25 and Semantic Search with RRF."
- **Level 1:** What is the mathematical formulation of BM25? How is it different from TF-IDF?
  - *Answer:* BM25 introduces **term frequency saturation** and **document length normalization**:
    $$\text{BM25}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
    $k_1$ controls term frequency saturation (prevents a repeated word from dominating), and $b$ controls length normalization penalty. TF-IDF increases linearly with term frequency without saturation.
- **Level 2:** What is RRF (Reciprocal Rank Fusion)? What is the formula?
  - *Answer:* 
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    where $k$ is a smoothing constant (typically 60) and $r_m(d)$ is the rank of document $d$ in retriever $m$.
- **Level 3:** Why not simply normalize BM25 scores (min-max) and add them to Cosine Similarity scores?
  - *Answer:* Min-max normalization depends heavily on extreme values (outliers) in a single query's result set. BM25 is unbounded $[0, \infty)$, while Cosine is $[-1, 1]$. Score distributions shift drastically per query, making linear weights unstable. RRF uses rank positions, making it immune to scale mismatch.
- **Level 4:** What is the "Lost in the Middle" problem in LLM context windows?
  - *Answer:* Research shows LLMs attend heavily to the beginning and end of long prompts, but frequently ignore or hallucinate information placed in the middle. RAG pipelines mitigate this by placing top reranked chunks at the very top or bottom of the injected context.
- **Level 5:** What is the difference between RAG and Fine-Tuning? When do you choose which?
  - *Answer:* 
    - **RAG:** Injecting dynamic, external, up-to-date knowledge; requires exact source citations; low compute cost.
    - **Fine-Tuning:** Teaching a model a new style, format, dialect, or domain-specific grammar; cannot easily delete or update outdated facts without retraining; prone to hallucination without grounding.

---

### 💬 "I used Groq API with Llama-3.3-70B and Pydantic validation."
- **Level 1:** What makes Groq so much faster than traditional NVIDIA GPUs?
  - *Answer:* Groq uses an **LPU (Language Processing Unit)** with a **Tensor Streaming Processor (TSP)** architecture. Unlike GPUs that rely on non-deterministic external high-bandwidth memory (HBM) and dynamic scheduling, Groq uses deterministic compiler-scheduled execution with huge on-chip SRAM, eliminating memory bandwidth bottlenecks.
- **Level 2:** What is the parameter size of Llama-3.3-70B? How much VRAM is needed to run it locally in FP16 vs INT4?
  - *Answer:* 70 Billion parameters. In FP16 (16-bit float = 2 bytes/param), it requires $\approx 140\text{ GB}$ VRAM. In 4-bit quantization (GGUF / AWQ / GPTQ $\approx 0.5\text{ bytes/param}$), it requires $\approx 38\text{--}42\text{ GB}$ VRAM.
- **Level 3:** What is Pydantic? How does it differ from dataclasses or raw dictionaries?
  - *Answer:* `dataclasses` only provide type hints without runtime enforcement. Pydantic enforces strict **runtime type validation and coercion** (e.g., converting a string `"25"` into an `int 25`), throws detailed `ValidationError` objects, and generates JSON Schema representations.

---

# 3. Computer Vision & Deep Learning (Glaucoma Project)

### 💬 "I used ResNet-18 for classification and U-Net for segmentation."
- **Level 1:** What problem does ResNet solve? What is the Vanishing Gradient problem?
  - *Answer:* As deep networks grow deeper, gradients shrink exponentially during backpropagation due to repeated multiplication of small weights/derivatives ($\frac{\partial L}{\partial w} \to 0$), causing early layers to stop learning. ResNet introduces **Residual Connections (Skip Connections)**:
    $$\mathcal{H}(x) = \mathcal{F}(x) + x$$
    During backpropagation: $\frac{\partial \mathcal{H}}{\partial x} = \frac{\partial \mathcal{F}}{\partial x} + 1$. The "$+1$" term ensures gradients flow directly backward without vanishing.
- **Level 2:** What is the architectural difference between ResNet-18 and ResNet-50?
  - *Answer:* ResNet-18 uses **Basic Blocks** (two $3 \times 3$ convolutions). ResNet-50 uses **Bottleneck Blocks** ($1 \times 1$ conv to reduce channels, $3 \times 3$ conv, $1 \times 1$ conv to restore channels) to reduce computational FLOPs in deeper layers.
- **Level 3:** Explain the U-Net architecture. Why are skip connections in U-Net different from ResNet?
  - *Answer:* U-Net is an **Encoder-Decoder** (contraction and expansion path) network with a bottleneck. In ResNet, skip connections perform **element-wise addition** ($+$) to preserve gradients. In U-Net, skip connections copy high-resolution spatial feature maps from the encoder and **concatenate** them along the channel dimension with the decoder features, restoring lost spatial details for precise pixel segmentation.
- **Level 4:** How does Grad-CAM work? Why is the final output passed through ReLU?
  - *Answer:* Grad-CAM computes gradients of target score $y^c$ w.r.t feature maps $A^k$ of the last conv layer, pools them into channel weights $\alpha_k^c$, and computes a weighted linear combination $\sum \alpha_k^c A^k$. **ReLU** is applied because we only care about features that have a *positive* influence on the target class score; negative activations belong to other classes.
- **Level 5:** Explain the difference between Precision, Recall, Sensitivity, Specificity, and ROC-AUC.
  - *Answer:*
    - $\text{Sensitivity} = \text{Recall} = \frac{TP}{TP + FN}$ (Fraction of sick patients correctly identified).
    - $\text{Specificity} = \frac{TN}{TN + FP}$ (Fraction of healthy patients correctly identified).
    - $\text{Precision} = \frac{TP}{TP + FP}$ (When model predicts sick, how often is it right?).
    - **ROC-AUC:** Area under the curve plotting True Positive Rate vs False Positive Rate ($1 - \text{Specificity}$) across all threshold cutoffs ($0.0 \to 1.0$).
- **Level 6:** What is the difference between Dice Loss and Cross-Entropy Loss?
  - *Answer:* Cross-Entropy evaluates classification at each pixel independently, which performs poorly when background pixels overwhelm foreground pixels (class imbalance). **Dice Loss** measures the spatial overlap (Intersection over Union) directly:
    $$\text{Dice Loss} = 1 - \frac{2 |X \cap Y|}{|X| + |Y|}$$

---

# 4. Backend Engineering (FastAPI, Node.js, Python, Concurrency)

### 💬 "I developed the backend using FastAPI."
- **Level 1:** What is the difference between WSGI and ASGI?
  - *Answer:* **WSGI (Web Server Gateway Interface)** (e.g., Flask, Django) is synchronous; each request blocks a thread until finished. **ASGI (Asynchronous Server Gateway Interface)** (e.g., FastAPI, Starlette) supports asynchronous Python coroutines (`async`/`await`), WebSockets, and long-lived connections on a single event loop thread.
- **Level 2:** What is the Python GIL (Global Interpreter Lock)? How does it affect FastAPI?
  - *Answer:* The GIL is a mutex that prevents multiple native OS threads from executing Python bytecodes simultaneously in CPython. For **I/O-bound tasks** (network calls, DB queries, LLM APIs), `asyncio` releases control and handles thousands of concurrent requests. For **CPU-bound tasks** (deep learning tensor math, image transformations), `asyncio` will block the event loop—these must be offloaded to `multiprocessing` or Celery workers.
- **Level 3:** What is the difference between `def endpoint()` and `async def endpoint()` in FastAPI?
  - *Answer:* If you declare `def`, FastAPI runs it in an external thread pool (Worker Thread). If you declare `async def`, FastAPI runs it directly on the main event loop thread. If you run blocking code (e.g., `time.sleep()` or heavy loops) inside `async def`, you freeze the entire server!

---

### 💬 "I have experience with Node.js and Express."
- **Level 1:** How does Node.js handle concurrency if it is single-threaded?
  - *Answer:* Node.js uses the **V8 engine** and **libuv Event Loop**. While JavaScript execution is single-threaded, asynchronous I/O operations (file system, DNS, crypto, network requests) are delegated to the libuv C++ background thread pool or kernel asynchronous APIs (epoll/kqueue).
- **Level 2:** What is Event-Driven Architecture? What is the Callback Queue vs Microtask Queue?
  - *Answer:* The **Microtask Queue** (Promises, `process.nextTick()`) has higher priority and executes immediately after the current operation before the Event Loop moves to the **Macrotask/Callback Queue** (`setTimeout`, `setInterval`, `setImmediate`).

---

# 5. Core CS Fundamentals (OOP, OS, Networks)

### 💬 Object-Oriented Programming (OOP)
- **Question 1:** Explain the 4 Pillars of OOP with real-life examples.
  - **Encapsulation:** Bundling data and methods together and restricting direct access (private variables with getters/setters).
  - **Abstraction:** Hiding complex implementation details and showing only the interface (e.g., Abstract Base Classes or Interfaces).
  - **Inheritance:** Deriving child classes from a parent class to promote code reuse.
  - **Polymorphism:** Ability of an object to take many forms (Compile-time / Method Overloading vs Runtime / Method Overriding).
- **Question 2:** What is the difference between Method Overloading and Method Overriding?
  - *Answer:* **Overloading** happens in the same class (same method name, different parameter types/count; resolved at compile time). **Overriding** happens in child classes (same method name, same signature, different implementation; resolved at runtime via dynamic dispatch).
- **Question 3:** What are SOLID principles?
  - **S:** Single Responsibility Principle (A class should have one and only one reason to change).
  - **O:** Open/Closed Principle (Open for extension, closed for modification).
  - **L:** Liskov Substitution Principle (Derived classes must be substitutable for their base classes).
  - **I:** Interface Segregation Principle (Clients should not be forced to depend on methods they don't use).
  - **D:** Dependency Inversion Principle (Depend on abstractions, not concretions).

---

### 💬 Operating Systems & Concurrency
- **Question 1:** What is the difference between a Process and a Thread?
  - *Answer:* A **Process** is an executing instance of a program with its own independent memory space (code, data, heap, stack). A **Thread** is a lightweight unit of execution within a process; all threads in a process share the same memory space (heap, global variables) but have their own private stacks and registers.
- **Question 2:** What is Deadlock? What are the 4 Coffman conditions required for Deadlock?
  - *Answer:* Deadlock is a state where a set of processes are blocked because each is holding a resource and waiting for another resource held by another process.
    1. **Mutual Exclusion:** Resources cannot be shared.
    2. **Hold and Wait:** A process holds at least one resource and is waiting for more.
    3. **No Preemption:** Resources cannot be forcibly taken away.
    4. **Circular Wait:** A closed chain of processes exists where each waits for a resource held by the next.
- **Question 3:** What is Virtual Memory and Paging? What is a Page Fault?
  - *Answer:* Virtual Memory provides an illusion of a large contiguous address space by mapping virtual addresses to physical RAM frames using **Page Tables**. A **Page Fault** occurs when a program tries to access a virtual memory page that is not currently loaded in physical RAM, triggering the OS to load it from disk (swap space).

---

### 💬 Computer Networks
- **Question 1:** What happens when you type `https://www.google.com` in your browser and hit Enter?
  - *Step 1:* Browser checks DNS cache (Browser -> OS -> Router -> ISP Recursive Resolver).
  - *Step 2:* DNS Resolution (Root -> TLD `.com` -> Authoritative Name Server) returns IP address.
  - *Step 3:* TCP 3-Way Handshake (`SYN` $\to$ `SYN-ACK` $\to$ `ACK`) establishes connection.
  - *Step 4:* TLS Handshake establishes encryption (certificate validation, key exchange via RSA/ECDHE, symmetric session keys).
  - *Step 5:* Browser sends `HTTP GET /` request.
  - *Step 6:* Server processes request and returns `HTTP 200 OK` with HTML.
  - *Step 7:* Browser engine parses HTML, constructs DOM/CSSOM tree, renders UI, and executes JS.
- **Question 2:** What is the difference between TCP and UDP?
  - *Answer:* **TCP** is connection-oriented, reliable (guaranteed delivery via ACKs, retransmissions), ordered, with flow/congestion control (used in HTTP, SSH, Email). **UDP** is connectionless, unreliable, unordered, with minimal overhead (used in Video Streaming, Gaming, DNS, VoIP).
- **Question 3:** What are the layers of the OSI model?
  - *Answer:* Physical $\to$ Data Link $\to$ Network $\to$ Transport $\to$ Session $\to$ Presentation $\to$ Application (Mnemonic: *Please Do Not Throw Sausage Pizza Away*).

---

# 6. DSA & Problem Solving (450+ LeetCode Solved)

### 💬 Live Coding & Algorithmic Traps
- **Trap 1:** *"You solved 450+ problems on LeetCode. What is the difference between QuickSort and MergeSort? Which one does Python's `sort()` use?"*
  - *Answer:* MergeSort is $O(N \log N)$ worst-case, stable, but requires $O(N)$ extra space. QuickSort is $O(N \log N)$ average-case, $O(N^2)$ worst-case, in-place $O(\log N)$ space, but unstable. Python uses **Timsort**, a hybrid stable sorting algorithm combining MergeSort and Insertion Sort with $O(N)$ best-case on partially sorted data.
- **Trap 2:** *"How do you detect a cycle in a Linked List without extra memory?"*
  - *Answer:* **Floyd's Tortoise and Hare Algorithm** (Fast and Slow pointers). Slow moves 1 step, Fast moves 2 steps. If they meet, there is a cycle. To find the cycle entry node: reset slow pointer to head; move both 1 step at a time until they meet. Time: $O(N)$, Space: $O(1)$.
- **Trap 3:** *"Explain Dynamic Programming. How is Top-Down different from Bottom-Up?"*
  - *Answer:* DP solves problems by breaking them into overlapping subproblems with optimal substructure. **Top-Down (Memoization):** Recursive approach storing results in a hashmap/array. **Bottom-Up (Tabulation):** Iterative approach filling a table starting from base cases, often allowing space optimization (e.g., $O(N)$ reduced to $O(1)$ space).

---

# 7. ServiceNow Certified System Administrator (CSA)

### 💬 "I see you have a ServiceNow CSA Certification."
- **Question 1:** What is ServiceNow, and what is the underlying architecture?
  - *Answer:* ServiceNow is an enterprise Cloud PaaS platform focusing on IT Service Management (ITSM), IT Operations (ITOM), and automated enterprise workflows. It uses a **multi-instance architecture** (each customer gets dedicated application and database instances, rather than shared multi-tenant database partitions).
- **Question 2:** What is the difference between a Business Rule, a Client Script, and a UI Policy?
  - *Answer:* 
    - **Client Script:** Runs on the client browser (JavaScript) during form load, submit, or field change (`onLoad`, `onSubmit`, `onChange`).
    - **UI Policy:** No-code/low-code client-side alternative to make fields mandatory, visible, or read-only dynamically.
    - **Business Rule:** Runs on the server side when records are displayed, inserted, updated, or deleted (`before`, `after`, `async`, `display`).
- **Question 3:** What is the CMDB (Configuration Management Database)?
  - *Answer:* A centralized database that tracks all IT Configuration Items (CIs) such as servers, software licenses, network devices, and business applications, along with their relational dependencies.
- **Question 4:** How does Access Control (ACL) work in ServiceNow?
  - *Answer:* ACL rules restrict access to data based on Roles, Conditions, and Scripts. Evaluation order: Table-level ACLs are checked first, followed by Field-level ACLs. Both must evaluate to `true` for a user to gain access.
