package org.example;

public class EmbeddingResult {
    double[] vector;
    long timeMs;

    public EmbeddingResult(double[] vector, long timeMs) {
        this.vector = vector;
        this.timeMs = timeMs;
    }
}