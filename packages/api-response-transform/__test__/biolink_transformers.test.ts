import jq_tf from "../built/transformers/jq_transformer";
import biolink_tf from "../src/transformers/biolink_transformer";
import fs from "fs";
import path from "path";
import { JSONDoc } from "../src/json_transform/types";

describe("test biolink transformer", () => {
  let response;
  let input;

  beforeEach(() => {
    const response_path = path.resolve(__dirname, "./data/biolink/response.json");
    response = JSON.parse(fs.readFileSync(response_path, { encoding: "utf8" }));
    const edge_path = path.resolve(__dirname, "./data/biolink/edge.json");
    const edge = JSON.parse(fs.readFileSync(edge_path, { encoding: "utf8" }));
    input = {
      response,
      edge,
    };
  });

  test("test biolink wrapper", async () => {
    const tf = new jq_tf(input, { type: "biolink" });
    const res = await tf.wrap(response);
    expect(res.associations[0].object.HGNC).toBe("10956");
    expect(res.associations[0].publications[0].id).toBe("21685912");
    expect(res.associations[1]).not.toHaveProperty("publications");
    expect(res.associations[1]).not.toHaveProperty("provided_by");
  });

  test("test biolink wrapper if no association field as root key", async () => {
    const tf = new jq_tf(input, { type: "biolink" });
    const res = await tf.wrap({ data: [] });
    expect(res).toEqual({ data: [] });
  });

  test("test biolink wrapper if no object id should be prefixed", async () => {
    const tf = new jq_tf(input, { type: "biolink" });
    const res = await tf.wrap({
      associations: [
        {
          object: {
            id: "MONDO:12345",
          },
        },
      ],
    });
    expect(res.associations[0].object.MONDO).toEqual("MONDO:12345");
  });

  test("test biolink wrapper if no object field present", async () => {
    const tf = new jq_tf(input, { type: "biolink" });
    const fake_response = {
      associations: [
        {
          object1: {
            id: "MONDO:12345",
          },
        },
      ],
    };
    const res = await tf.wrap(fake_response);
    expect(res).toEqual(fake_response);
  });

  test("test biolink wrapper if no object.id field present", async () => {
    const tf = new jq_tf(input, { type: "biolink" });
    const fake_response = {
      associations: [
        {
          object: {
            id1: "MONDO:12345",
          },
        },
      ],
    };
    const res = await tf.wrap(fake_response);
    expect(res).toEqual(fake_response);
  });

  test("test biolink jsonTransform function", async () => {
    const tf = new jq_tf(input, { type: "biolink" });
    const wrapped_response = await tf.wrap(response);
    const res: JSONDoc = tf.jsonTransform(wrapped_response);
    expect(res).toHaveProperty("related_to");
    expect(res.related_to[0].HGNC).toEqual("10956");
    expect(res.related_to[0].pubmed[0]).toEqual("21685912");
    expect(res.related_to[0].relation).toEqual("contributes to condition");
    expect(res.related_to[0].source[0]).toEqual("https://archive.monarchinitiative.org/#gwascatalog");
    expect(res.related_to[0].taxid).toEqual("NCBITaxon:9606");
  });
});
