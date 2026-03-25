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
                "all-minilm",
                "bge-m3"
        );

        for (String model : models) {
            runEvaluation(model, query, stored);
        }
    }

    private static void runEvaluation(String model, String query, List<String> stored) {

        System.out.println("\n===== MODEL: " + model + " =====");

        // 🔹 Query embedding
        EmbeddingResult queryRes =
                OllamaClient.getSingleEmbedding(model, query);

        System.out.println("Embedding Time (" + query + "): " + queryRes.timeMs + " ms");

        List<ResultRow> results = new ArrayList<>();

        for (String s : stored) {

            EmbeddingResult storedRes =
                    OllamaClient.getSingleEmbedding(model, s);

            long simStart = System.nanoTime();

            double sim = SimilarityUtils.cosineSimilarity(
                    queryRes.vector,
                    storedRes.vector
            );

            long simEnd = System.nanoTime();

            long simTime = (simEnd - simStart) / 1_000; // microseconds

            results.add(new ResultRow(s, sim, storedRes.timeMs, simTime));
        }

        // Sort DESC
        results.sort((a, b) -> Double.compare(b.score, a.score));

        // 🔹 Print embedding time for top 3
        for (int i = 0; i < Math.min(3, results.size()); i++) {
            ResultRow r = results.get(i);
            System.out.println("Embedding Time (" + r.text + "): " + r.embeddingTime + " ms");
        }

        // 🔹 Print matches
        System.out.println("\nTop Matches:");

        for (int i = 0; i < Math.min(3, results.size()); i++) {
            ResultRow r = results.get(i);

            System.out.printf(
                    "%d. Score: %.4f (sim: %d µs)\n   \"%s\" vs \"%s\"\n\n",
                    i + 1,
                    r.score,
                    r.simTime,
                    query,
                    r.text
            );
        }
    }
}