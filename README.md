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

public class SentenceEmbeddingTranslator implements Translator<String, float[]> {

    private HuggingFaceTokenizer tokenizer;

    @Override
    public void prepare(TranslatorContext ctx) throws Exception {
        tokenizer = HuggingFaceTokenizer.newInstance(
                ctx.getModel().getModelPath()
        );
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) {

        String text = "query: " + input; // REQUIRED for E5

        Encoding enc = tokenizer.encode(text);

        NDArray inputIds = ctx.getNDManager()
                .create(enc.getIds())
                .expandDims(0);

        NDArray attentionMask = ctx.getNDManager()
                .create(enc.getAttentionMask())
                .expandDims(0);

        return new NDList(inputIds, attentionMask);
    }

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {

        // output[0] = token embeddings [1, seq_len, hidden]
        NDArray tokenEmbeddings = list.get(0);

        // Mean pooling
        NDArray pooled = tokenEmbeddings.mean(new int[]{1});

        return pooled.toFloatArray();
    }
}

public class LocalE5LargeEmbeddingClient implements LlmClient, AutoCloseable {

    private final ZooModel<String, float[]> model;
    private final Predictor<String, float[]> predictor;

    public LocalE5LargeEmbeddingClient(Path modelDir) throws Exception {

        Criteria<String, float[]> criteria =
                Criteria.builder()
                        .setTypes(String.class, float[].class)
                        .optModelPath(modelDir)
                        .optTranslator(new SentenceEmbeddingTranslator())
                        .optEngine("PyTorch")
                        .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();
    }

    @Override
    public float[] generateEmbeddings(String text) {
        try {
            return predictor.predict(text);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void close() throws Exception {
        predictor.close();
        model.close();
    }
}


```
