# Agentic Memory with Oracle AI Vector Search

**Semantic memory for conversation agents using Oracle Autonomous Database**

Last year, I worked on a conversation agent for a loan optimization system (can't share specifics due to data protection). The agent would call borrowers, have conversations in multiple languages, and record summaries. The challenge was finding relevant information from past conversations, as keyword search didn't work well.

This project explores how **semantic memory with vector search** could have solved that problem, using Oracle Autonomous Database's native VECTOR capabilities.

### What This Demonstrates

A memory system for AI agents that can:
1. Store conversations as semantic embeddings.
2. Search by meaning, not keywords.
3. Retrieve relevant context for better responses.
4. Scale to production workloads.

**Example:**
```
Stored: "User prefers email notifications over SMS"
Query: "What's the user's communication preference?"
Result: Finds the email preference (0.847 similarity)
```

Even with different wording, semantic search finds relevant information.

## Why Oracle?

**Technical reasons:**
- **Native VECTOR type**: No extensions needed like pgvector
- **Hybrid SQL + vector**: Combine semantic search with business logic in one query
- **In-memory indexes**: Sub-30ms search on thousands of vectors
- **ACID transactions**: Important for financial/regulated data

**Practical reasons:**
- Already using Oracle for main application.
- Local embeddings (data protection, so nothing leaves server).
- Free tier available for experimentation.
- Enterprise support and reliability.

**Alternative considered:**
- Pinecone/Weaviate: Separate service to manage.
- PostgreSQL + pgvector: Extension, not native.
- ChromaDB: More suited for experimentation than regulated production environments.

Oracle made sense for regulated industries where data protection and reliability matter.

## Quick Start

### Prerequisites

- Python 3.9+
- Oracle Autonomous Database (free tier)
- Wallet downloaded and extracted
- Oracle Instant Client

### Installation

```bash
# Clone repository
git clone https://github.com/srinidhi-sat/agent-memory.git

# Install dependencies
pip install sentence-transformers oracledb numpy

# Download Oracle Instant Client
# https://www.oracle.com/database/technologies/instant-client/downloads.html
```

### Configuration

1. Create Oracle Cloud Account:
    - Go to Oracle Cloud Free Tier.
    - Sign up with email.
2. Create Autonomous Database:
    - Deployment type: "Shared Infrastructure"
    - Always Free: Check this box
    - Database name: e.g., AGENTMEM
    - Workload type: Transaction Processing
    - Password: Set admin password (Add as DATABASE_PASSWORD)
3. Configure Network Access:
    - After database creation, go to "Database Details"
    - Click "Edit Access Rules".
    - Add your IP address.
4. Download Wallet:
    - On Database Details page, click "DB Connection"
    - Click "Download Wallet"
    - Set wallet password (optional but recommended)
    - Download zip file
    - Extract to a local folder (Add as WALLET_DIR)


### Thick Mode Configuration
This example uses Oracle Thick mode.

1. Why Thick mode?
    - Required for full VECTOR support in Oracle 26ai
    - Enables Oracle-specific advanced features
    - Uses native C libraries for stable database operations

2. Install Oracle Instant Client
    - Download from Oracle Instant Client Downloads
    - Choose a version compatible with your database
    - Extract to folder
    - Add the path as INSTANT_CLIENT_DIR.


### Blog

To read the full blog, open `blogpost.md`.


### Run

1. Open `agent_memory.ipynb`.
2. Update configuration variables:
    - WALLET_DIR
    - DATABASE_PASSWORD
    - INSTANT_CLIENT_DIR
3. Run the notebook from top to bottom.
4. The notebook will:
    - Create the memory table
    - Insert sample memory records
    - Execute vector similarity search
    - Generate an example agent response


## Project Structure

```
main
├── agent_memory.ipynb           # Main demo notebook
├── blogpost.md                  # Detailed technical blog
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## How It Works

### 1. Database Schema

```sql
CREATE TABLE memories (
    id NUMBER PRIMARY KEY,
    text CLOB,                      -- Memory content
    embedding VECTOR(384, FLOAT32), -- Semantic representation
    memory_type VARCHAR2(50),       -- Category
    created_at TIMESTAMP
);
```

Key feature: Oracle's native `VECTOR(384, FLOAT32)` type.

### 2. Storing Memories

```python
# Generate embedding locally (data protection)
embedding = model.encode(text)

# Store in Oracle
cursor.execute("""
    INSERT INTO memories (text, embedding, memory_type)
    VALUES (:1, TO_VECTOR(:2), :3)
""", [text, str(embedding.tolist()), memory_type])
```

### 3. Semantic Search

```python
# Search by meaning
query_embedding = model.encode(query)

cursor.execute("""
    SELECT text,
    (1 - VECTOR_DISTANCE(embedding, TO_VECTOR(:1), COSINE)) as similarity
    FROM memories
    ORDER BY similarity DESC
    FETCH FIRST 5 ROWS ONLY
""", [str(query_embedding.tolist())])
```

### 4. RAG Pattern

```python
# 1. Search relevant memories
results = memory.search(user_query, top_k=3)

# 2. Build LLM prompt with context
context = "\n".join([r['text'] for r in results])
prompt = f"Context: {context}\n\nQuestion: {user_query}"

# 3. Call LLM (Claude, GPT-4, etc.)
response = llm.generate(prompt)
```

## Use Cases

### Customer Support
```python
memory.store("User reported slow upload speeds", "issue")
# Later: search("technical problems"), finds the issue
```

### Conversation Agents (Loan System Example)
```python
memory.store("Borrower promised payment by 15th", "repayment_promise")
memory.store("Mentioned medical emergency", "context")
# Before call: search("previous conversations"), agent has context
```

### Multi-language Systems
```python
# Using multilingual models, search in English, find Hindi conversations
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

## Performance

Benchmarks were run locally using the Oracle Always Free tier with a small memory dataset.

| Operation               | Latency | Scale                   |
|-------------------------|---------|-------------------------|
| Store memory            | ~50ms   | Per memory              |
| Search (1K memories)    | ~15ms   | 5 results               |
| Search (10K memories)   | ~30ms   | 5 results               |
| Search (100K memories)  | ~50ms   | 5 results (with index)  |

Fast enough for real-time applications.

## Production Considerations

For production deployment for your use case, add:

### 1. User Isolation
```sql
ALTER TABLE memories ADD borrower_id VARCHAR2(100);
WHERE borrower_id = :id AND VECTOR_DISTANCE(...) < 0.3
```

### 2. Hybrid Queries
```sql
-- Combine business logic with semantic search
WHERE risk_score > 0.7 
AND geography = 'Maharashtra'
AND VECTOR_DISTANCE(embedding, :query, COSINE) < 0.3
```

### 3. Vector Indexes
```sql
CREATE VECTOR INDEX memory_idx ON memories(embedding)
ORGANIZATION INMEMORY NEIGHBOR GRAPH
WITH TARGET ACCURACY 95;
```

### 4. Memory Management
- Consolidate similar memories.
- Clean old data.
- Implement importance scoring.

## Technical Details

**Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions)
- Fast (~50ms per text).
- Good semantic understanding.
- Runs locally (data protection).

**Oracle VECTOR_DISTANCE:**
- Supports COSINE, EUCLIDEAN, DOT metrics.
- Leverages in-memory indexes.
- Sub-100ms at scale.

**Why COSINE distance:**
- Normalized (0-1 range).
- Direction matters, magnitude doesn't.
- Standard for semantic similarity.

## Limitations & Trade-offs

**Current implementation:**
- Single user (no isolation).
- English only (can be extended with multilingual models).
- No memory consolidation.
- No importance scoring.

**Design decisions:**
- 384 dimensions (could use 768 for more nuance, slower).
- Local embeddings (vs API-based like OpenAI- chose data protection).

## What I Learned

1. **Oracle's vector search is solid**: Fast, reliable, integrates with SQL.
2. **Hybrid queries are powerful**: Semantic + filters in one query is genuinely useful.
3. **Local embeddings matter**: For regulated industries, data can't leave server.
4. **Memory types are essential**: Categories improve search accuracy.
5. **384 dimensions is enough**: Bigger isn't always better.

## Resources

- [Blogpost Link](https://github.com/srinidhi-sat/agent-memory/blob/main/blogpost.md)
- [Jupter Notebook Link](https://github.com/srinidhi-sat/agent-memory/blob/main/agent_memory.ipynb)
- [Requirements](https://github.com/srinidhi-sat/agent-memory/blob/main/requirements.txt)
- [Oracle 26ai Vector Search Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/)
- [Sentence Transformers](https://www.sbert.net/)
- [python-oracledb](https://python-oracledb.readthedocs.io/)
- [Python 3.9+](https://www.python.org/downloads/)
- [Oracle Always Free Tier](https://www.oracle.com/cloud/free/)

## Contributing

This is an exploration project, but feedback and suggestions welcome. Open an issue or PR.

## License

MIT

## Contact

Questions or feedback? Open an issue on GitHub.


*This project was built to explore semantic memory for AI agents, based on challenges encountered in a loan conversation system. The implementation demonstrates Oracle 26ai's vector capabilities with a practical, production-oriented approach.*
