import base_tf from "../src/transformers/transformer";
import fs from "fs";
import path from "path";
import { JSONDoc } from "../src/json_transform/types";

describe("test base transformer", () => {
  let response;
  let input;

  beforeEach(() => {
    const response_path = path.resolve(__dirname, "./data/ols/response.json");
    response = JSON.parse(fs.readFileSync(response_path, { encoding: "utf8" }));
    const edge_path = path.resolve(__dirname, "./data/ols/edge.json");
    const edge = JSON.parse(fs.readFileSync(edge_path, { encoding: "utf8" }));
    input = {
      response,
      edge,
    };
  });

  test("Test pairInputWithAPIResponse function", async () => {
    const tf = new base_tf(input, {});
    const res = await tf.pairCurieWithAPIResponse();
    expect(res).toHaveProperty("DOID:9562");
    expect(res["DOID:9562"]).toHaveLength(1);
  });

  test("Test wrap function if response is not an array", () => {
    const tf = new base_tf(input, {});
    const res = tf.wrap(response);
    expect(res).toHaveProperty("_embedded");
  });

  test("Test wrap function if response is an array", () => {
    const tf = new base_tf(input, {});
    const fake = ["1"];
    const res = tf.wrap(fake);
    expect(res).toHaveProperty("data");
    expect(res.data).toEqual(["1"]);
  });

  test("Test jsonTransform function", () => {
    const tf = new base_tf(input, {});
    const res: JSONDoc = tf.jsonTransform(response);
    expect(res).toHaveProperty("has_subclass");
    expect(res.has_subclass[0]).toHaveProperty("DOID");
    expect(res.has_subclass[0].DOID).toEqual("DOID:0110596");
    expect(res.has_subclass[0].name).toEqual("primary ciliary dyskinesia 21");
    expect(res.has_subclass[0].description).toEqual([
      "A primary ciliary dyskinesia that is characterized by autosomal recessive inheritance with a missing Nexin link, infantile onset of chronic sinopulmonary infections, and has_material_basis_in homozygous mutation in the DRC1 gene on chromosome 2p23.",
    ]);
  });

  test("Test _updatePublications function if pubmed id is prefixed", () => {
    const tf = new base_tf(input, {});
    const fake = {
      ref_pmid: "PMID:1233",
    };
    const res = tf._updatePublications(fake);
    expect(res).not.toHaveProperty("ref_pmid");
    expect(res.publications).toEqual(["PMID:1233"]);
  });

  test("Test _updatePublications function if pubmed id is NOT prefixed", () => {
    const tf = new base_tf(input, {});
    const fake = {
      ref_pmid: 1233,
    };
    const res = tf._updatePublications(fake);
    expect(res).not.toHaveProperty("ref_pmid");
    expect(res.publications).toEqual(["PMID:1233"]);
  });

  test("Test _updatePublications function if pmc id is prefixed", () => {
    const tf = new base_tf(input, {});
    const fake = {
      ref_pmcid: "PMCID:1233",
    };
    const res = tf._updatePublications(fake);
    expect(res).not.toHaveProperty("ref_pmcid");
    expect(res.publications).toEqual(["PMCID:1233"]);
  });

  test("Test _updatePublications function if pmc id is NOT prefixed", () => {
    const tf = new base_tf(input, {});
    const fake = {
      ref_pmcid: 123,
    };
    const res = tf._updatePublications(fake);
    expect(res).not.toHaveProperty("ref_pmcid");
    expect(res.publications).toEqual(["PMCID:123"]);
  });

  test("Test extractObjectIDs function if output id type not in result", () => {
    const tf = new base_tf(input, {});
    const fake = {
      kk: 1,
    };
    const res = tf.extractObjectIDs(fake);
    expect(res).toEqual([]);
  });

  test("Test extractObjectIDs function if output id type is in result", () => {
    const tf = new base_tf(input, {});
    const fake = {
      DOID: 1,
    };
    const res = tf.extractObjectIDs(fake);
    expect(res).toEqual(["DOID:1"]);
  });

  test("Test formatRecords function if result is empty", async () => {
    const tf = new base_tf(input, {});
    const fake = {};
    const res = await tf.formatRecords("NCBIGene:1017", fake);
    expect(res).toEqual([]);
  });
});
