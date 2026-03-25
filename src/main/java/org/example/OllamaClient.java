package org.example;

import okhttp3.*;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class OllamaClient {

    private static final String URL = "http://localhost:11434/api/embed";
    private static final OkHttpClient CLIENT = new OkHttpClient();

    private OllamaClient() {}

    public static List<double[]> getEmbeddings(String model, List<String> inputs) {

        List<double[]> result = new ArrayList<>();

        try {
            JSONObject json = new JSONObject();
            json.put("model", model);
            json.put("input", inputs);

            Request request = new Request.Builder()
                    .url(URL)
                    .post(RequestBody.create(
                            json.toJSONString(),
                            MediaType.parse("application/json")
                    ))
                    .build();

            Response response = CLIENT.newCall(request).execute();

            if (!response.isSuccessful()) {
                throw new RuntimeException("API call failed: " + response);
            }

            String responseBody = response.body().string();

            JSONParser parser = new JSONParser();
            JSONObject responseJson = (JSONObject) parser.parse(responseBody);

            List<List<Double>> embeddings =
                    (List<List<Double>>) responseJson.get("embeddings");

            for (List<Double> emb : embeddings) {
                double[] vector = new double[emb.size()];
                for (int i = 0; i < emb.size(); i++) {
                    vector[i] = emb.get(i);
                }
                result.add(vector);
            }

        } catch (Exception e) {
            System.err.println("Error while fetching embeddings for model: " + model);
            e.printStackTrace();
        }

        return result;
    }

    // 🔹 Single embedding with time
    public static EmbeddingResult getSingleEmbedding(String model, String input) {

        long start = System.currentTimeMillis();

        List<double[]> result =
                getEmbeddings(model, Collections.singletonList(input));

        long end = System.currentTimeMillis();

        return new EmbeddingResult(result.get(0), (end - start));
    }
}