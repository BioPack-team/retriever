import fs from "fs";
import path from "path";
import semmed_tf from "../src/transformers/semmed_transformer";

describe("test semmed transformer", () => {
  let response, input;

  beforeEach(() => {
    const response_path = path.resolve(__dirname, "./data/semmedgene/response.json");
    response = JSON.parse(fs.readFileSync(response_path, { encoding: "utf8" }));
    const edge_path = path.resolve(__dirname, "./data/semmedgene/edge.json");
    const edge = JSON.parse(fs.readFileSync(edge_path, { encoding: "utf8" }));
    input = {
      response,
      edge,
    };
  });

  test("test semmed pairCurieWithAPIResponse", async () => {
    const tf = new semmed_tf(input, {});
    const res = await tf.pairCurieWithAPIResponse();
    expect(res["UMLS:C1332823"][0]["umls"]).toBe("C1332823");
    expect(res).toHaveProperty("UMLS:C1332823");
    expect(res["UMLS:123"]).toBeUndefined();
  });

  test("test wrapper", () => {
    const tf = new semmed_tf(input, {});
    const res = tf.wrap(input.response[0]);
    expect(res).toHaveProperty("positively_regulates");
  });

  test("test json transform", () => {
    const tf = new semmed_tf(input, {});
    const res = tf.jsonTransform(input.response[0]);
    expect(res).toEqual(input.response[0]);
  });

  test("add edge info", async () => {
    const tf = new semmed_tf(input, {});
    const res = await tf.pairCurieWithAPIResponse();
    let rec = res["UMLS:C1332823"][0];
    rec = tf.wrap(rec);
    const result = await tf.formatRecords("UMLS:C1332823", rec["positively_regulates"][0]);
    expect(result[0]).toHaveProperty("association");
    expect(result[0].api).toBe("SEMMED Gene API");
  });

  test("test main function transform", async () => {
    const tf = new semmed_tf(input, {});
    const res = await tf.transform();
    expect(res[0]).not.toHaveProperty("UMLS");
    expect(res[0]).not.toHaveProperty("@type");
    expect(res[0]).toHaveProperty("association");
    expect(res[0]).toHaveProperty("subject");
    expect(res.slice(-1)[0]).toHaveProperty("subject");
    expect(res.length).toBeGreaterThan(30);
  });
});
