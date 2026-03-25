package org.example;

import java.util.*;

public class EvaluationRunner {

    public static void main(String[] args) {

        String query = "i love java";

        List<String> stored = Arrays.asList(
                "i like java",
                "java is great",
                "python developer",
                "i enjoy coding",
                "java backend developer",
                "spring boot project",
                "i hate java",
                "javascript frontend",
                "machine learning engineer",
                "coffee is love"
        );

        List<String> models = Arrays.asList(
//                "qwen3-embedding:0.6b",
//                "nomic-embed-text-v2-moe",
//                "all-minilm",
                "bge-m3"
        );

        for (String model : models) {

            System.out.println("\n===== MODEL: " + model + " =====");

            runEvaluation(model, query, stored);
        }
    }

    private static void runEvaluation(String model, String query, List<String> stored) {

        // Combine query + stored for single API call
        List<String> inputs = new ArrayList<>();
        inputs.add(query);
        inputs.addAll(stored);

        long start = System.currentTimeMillis();

        List<double[]> embeddings =
                OllamaClient.getEmbeddings(model, inputs);

        long end = System.currentTimeMillis();

        double[] queryVector = embeddings.get(0);

        List<SimilarityResult> results = new ArrayList<>();

        for (int i = 1; i < embeddings.size(); i++) {

            double sim = SimilarityUtils.cosineSimilarity(
                    queryVector,
                    embeddings.get(i)
            );

            results.add(new SimilarityResult(stored.get(i - 1), sim));
        }

        // Sort descending
        results.sort((a, b) -> Double.compare(b.score, a.score));

        // Print top 3
        printTopK(results, query, model, (end - start), 3);
    }

    private static void printTopK(List<SimilarityResult> results,
                                  String query,
                                  String model,
                                  long time,
                                  int k) {

        for (int i = 0; i < Math.min(k, results.size()); i++) {

            SimilarityResult r = results.get(i);

            System.out.printf(
                    "%d. Score: %.4f | Time: %d ms\n   \"%s\" vs \"%s\"\n",
                    i + 1,
                    r.score,
                    time,
                    query,
                    r.text
            );
        }
    }
}