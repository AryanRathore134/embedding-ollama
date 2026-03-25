package org.example;

import java.util.Arrays;
import java.util.List;

public class EvaluationRunner {

    public static void main(String[] args) {

        List<String> inputs = Arrays.asList(
                "How are you?",
                "How are you doing?",
                "Who are you?"
        );

        List<String> models = Arrays.asList(
                "qwen3-embedding:0.6b",
                "nomic-embed-text-v2-moe",
                "all-minilm"
        );

        for (String model : models) {

            System.out.println("\n===== MODEL: " + model + " =====");

            List<double[]> embeddings =
                    OllamaClient.getEmbeddings(model, inputs);

            evaluate(embeddings);
        }
    }

    private static void evaluate(List<double[]> embeddings) {

        for (int i = 0; i < embeddings.size(); i++) {
            for (int j = i + 1; j < embeddings.size(); j++) {

                double sim = SimilarityUtils.cosineSimilarity(
                        embeddings.get(i),
                        embeddings.get(j)
                );

                System.out.printf("Q%d vs Q%d → %.4f%n", i, j, sim);
            }
        }
    }
}