```python

[tool.poetry]
name = "embedding-service"
version = "0.1.0"
description = "Offline embedding and cosine similarity service"
authors = ["you"]

[tool.poetry.dependencies]
python = ">=3.9,<3.13"

fastapi = "^0.110.0"
uvicorn = "^0.27.0"
sentence-transformers = "^2.7.0"
torch = "^2.2.0"
numpy = "^1.26.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

pip install \
  fastapi==0.110.0 \
  uvicorn==0.27.1 \
  torch==2.2.1 \
  sentence-transformers==2.7.0 \
  numpy==1.26.4

poetry install



from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# ======================================================
# 🔒 HARD OFFLINE MODE
# ======================================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ======================================================
# 🔧 CONFIG
# ======================================================
MODEL_PATH = r"C:\models\e5-large"   # 👈 change if needed
DEVICE = "cpu"

# ======================================================
# 🚀 APP
# ======================================================
app = FastAPI(title="Embedding & Cosine Service")

print(f"Loading model from local path: {MODEL_PATH}")

model = SentenceTransformer(
    MODEL_PATH,
    device=DEVICE
)

# ======================================================
# 📦 DTOs
# ======================================================
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

class CosineRequest(BaseModel):
    text1: str
    text2: str

class CosineResponse(BaseModel):
    cosineScore: float

# ======================================================
# 🧮 UTILS
# ======================================================
def embed_text(text: str) -> np.ndarray:
    """
    Embed text as a QUESTION.
    """
    text = "query: " + text
    return model.encode(
        text,
        normalize_embeddings=True
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two normalized vectors.
    """
    return float(np.dot(a, b))

# ======================================================
# 🔌 APIs
# ======================================================
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    vec = embed_text(req.text)
    return EmbedResponse(embedding=vec.tolist())


@app.post("/cosine", response_model=CosineResponse)
def cosine(req: CosineRequest):
    v1 = embed_text(req.text1)
    v2 = embed_text(req.text2)

    score = cosine_similarity(v1, v2)
    return CosineResponse(cosineScore=score)


@app.get("/health")
def health():
    return {"status": "ok"}


poetry run uvicorn embedding_service.app:app --host 0.0.0.0 --port 8000



```






```java
@SpringBootApplication
public class Application {

    static {
        // ===== HARD OFFLINE =====
        System.setProperty("DJL_OFFLINE", "true");
        System.setProperty("ai.djl.offline", "true");
        System.setProperty("ai.djl.repository.disable", "true");

        // ===== FORCE WINDOWS PATHS =====
        System.setProperty("ai.djl.cache.dir", "C:\\djl-cache");
        System.setProperty("ai.djl.tmpdir", "C:\\djl-tmp");
        System.setProperty("java.io.tmpdir", "C:\\djl-tmp");

        // ===== PYTORCH INDEX KILL SWITCHES =====
        System.setProperty("ai.djl.pytorch.disableIndex", "true");
        System.setProperty("ai.djl.pytorch.use_index", "false");
        System.setProperty("PYTORCH_INDEX_DISABLE", "true");

        // ===== HF PARANOIA =====
        System.setProperty("HF_HUB_OFFLINE", "1");
        System.setProperty("HF_HUB_DISABLE_TELEMETRY", "1");
    }

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}



```


# CosineSimilarityUtil 

```java

/**
 * Minimal cosine similarity utility for float vectors
 */
public final class CosineSimilarityUtil {

    private CosineSimilarityUtil() {}

    public static double cosine(float[] a, float[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Vectors must not be null");
        }
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vector length mismatch: " + a.length + " vs " + b.length
            );
        }

        double dot = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0 || normB == 0) {
            return 0.0;
        }

        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /** Normalize vector in-place (recommended) */
    public static void normalize(float[] v) {
        double norm = 0.0;
        for (float x : v) {
            norm += x * x;
        }
        norm = Math.sqrt(norm);
        if (norm == 0.0) return;

        for (int i = 0; i < v.length; i++) {
            v[i] /= norm;
        }
    }
}


```


# Pom.xml

```xml
<dependencies>

    <!-- DJL API -->
    <dependency>
        <groupId>ai.djl</groupId>
        <artifactId>api</artifactId>
        <version>0.26.0</version>
    </dependency>

    <!-- PyTorch engine -->
    <dependency>
        <groupId>ai.djl.pytorch</groupId>
        <artifactId>pytorch-engine</artifactId>
        <version>0.26.0</version>
    </dependency>

    <!-- Native CPU backend -->
    <dependency>
        <groupId>ai.djl.pytorch</groupId>
        <artifactId>pytorch-native-cpu</artifactId>
        <version>2.1.2</version>
        <scope>runtime</scope>
    </dependency>

</dependencies>

```


```java

public class SentenceEmbeddingTranslator
        implements Translator<String, float[]> {

    private HuggingFaceTokenizer tokenizer;

    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        tokenizer = HuggingFaceTokenizer.newInstance(
                ctx.getModel().getModelPath()
        );
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {

        // EVERYTHING is a question
        String text = "query: " + input;

        Encoding enc = tokenizer.encode(text);
        NDManager mgr = ctx.getNDManager();

        NDArray inputIds = mgr.create(enc.getIds()).expandDims(0);
        NDArray attentionMask = mgr.create(enc.getAttentionMask()).expandDims(0);

        return new NDList(inputIds, attentionMask);
    }

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {

        // [1, seq_len, hidden]
        NDArray tokenEmbeddings = list.get(0);

        // Mean pooling
        NDArray pooled = tokenEmbeddings.mean(new int[]{1});

        return pooled.toFloatArray();
    }
}



public class LocalE5LargeEmbeddingClient
        implements LlmClient, AutoCloseable {

    private final ZooModel<String, float[]> model;
    private final Predictor<String, float[]> predictor;

    public LocalE5LargeEmbeddingClient(Path modelDir) throws Exception {

        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optTranslator(new SentenceEmbeddingTranslator())
                        .optEngine("PyTorch")
                        .optOption("offline", "true")
                        .optOption("load_on_demand", "false")
                        .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    @Override
    public float[] generateEmbeddings(String text) {
        try {
            float[] vec = predictor.predict(text);
            CosineSimilarityUtil.normalize(vec);
            return vec;
        } catch (Exception e) {
            throw new RuntimeException("Embedding failed", e);
        }
    }

    @Override
    public void close() throws Exception {
        predictor.close();
        model.close();
    }
}


package com.example.embedding;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Service;

import java.nio.file.Path;

@Service
public class EmbeddingService {

    private LocalE5LargeEmbeddingClient client;

    @PostConstruct
    public void init() throws Exception {
        client = new LocalE5LargeEmbeddingClient(
                Path.of("models/e5-large")
        );
    }

    public float[] embed(String text) {
        return client.generateEmbeddings(text);
    }

    @PreDestroy
    public void shutdown() throws Exception {
        client.close();
    }
}


    @PostMapping("/embed")
    public float[] embed(@RequestBody EmbedRequest req) {
        return embeddingService.embed(req.text());
    }

    @PostMapping("/similarity")
    public SimilarityResponse similarity(
            @RequestBody SimilarityRequest req) {

        float[] a = embeddingService.embed(req.text1());
        float[] b = embeddingService.embed(req.text2());

        double score = CosineSimilarityUtil.cosine(a, b);
        return new SimilarityResponse(score);
    }


public record EmbedRequest(String text) {}

public record SimilarityRequest(String text1, String text2) {}

public record SimilarityResponse(double cosineScore) {}

    static {
        // 🔒 HARD OFFLINE MODE
        System.setProperty("DJL_OFFLINE", "true");
        System.setProperty("ai.djl.offline", "true");

        // 🔒 Disable HuggingFace hub completely
        System.setProperty("HF_HUB_OFFLINE", "1");
        System.setProperty("HF_HUB_DISABLE_TELEMETRY", "1");

        // 🔒 Prevent any repo resolution
        System.setProperty("ai.djl.repository.disable", "true");
    }

```


```json
{
  "text": "how do I reset my password"
}


{
  "text1": "how do I reset my password",
  "text2": "I forgot my password, how can I change it?"
}

{
  "text1": "how do I reset my password",
  "text2": "how can I change my account password?"
}

{
  "text1": "Can I track my move request?",
  "text2": "How do I check the status of my move?"
}

{
  "text1": "Can I track my move request?",
  "text2": "Status updates are available in the request history."
}


```
