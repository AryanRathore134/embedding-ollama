package org.example;

public class ResultRow {
    String text;
    double score;
    long embeddingTime;
    long simTime;

    public ResultRow(String text, double score, long embeddingTime, long simTime) {
        this.text = text;
        this.score = score;
        this.embeddingTime = embeddingTime;
        this.simTime = simTime;
    }
}