import API from "../../../src/parser/index";
import fs from "fs";
import path from "path";
import { SmartAPISpec, APIClass, ParsedAPIMetadataObject } from "../../../src/parser/types";

describe("test API parser", () => {
  let mygene: APIClass;
  let metadata: ParsedAPIMetadataObject;
  let ops;
  beforeEach(() => {
    const mygene_doc = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, "../../data/mygene.json"), { encoding: "utf8" }),
    ) as SmartAPISpec;
    mygene = new API(mygene_doc);
    metadata = mygene.metadata;
  });

  test("test parse API name", () => {
    expect(metadata.title).toBe("MyGene.info API");
  });

  test("test parse API Tags", () => {
    expect(metadata.tags).toContain("biothings");
  });

  test("test parse server url", () => {
    expect(metadata.url).toBe("https://mygene.info/v3");
  });

  test("test parse component", () => {
    expect(metadata.components).not.toBeUndefined();
  });

  test("test fetch meta data", () => {
    expect(metadata.title).toBe("MyGene.info API");
    expect(metadata.tags).toContain("biothings");
    expect(metadata.url).toBe("https://mygene.info/v3");
    expect(metadata.components).not.toBeUndefined();
  });

  test("test fetch all operations", () => {
    const ops = metadata.operations;
    expect(ops[0].association.api_name).toBe("MyGene.info API");
  });
});

describe("test API parser which is already dereferenced", () => {
  let opentarget: APIClass;
  let metadata: ParsedAPIMetadataObject;
  let ops;
  beforeEach(() => {
    const smartapi_spec = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, "../../data/opentarget.json"), { encoding: "utf8" }),
    ) as SmartAPISpec;
    opentarget = new API(smartapi_spec);
    metadata = opentarget.metadata;
  });

  test("test parse component", () => {
    expect(metadata.components).toBeUndefined();
  });

  test("test fetch meta data", () => {
    expect(metadata.title).toBe("OpenTarget API");
    expect(metadata.tags).toContain("translator");
    expect(metadata.url).toBe("https://platform-api.opentargets.io/v3");
    expect(metadata.components).toBeUndefined();
  });

  test("test fetch all operations", () => {
    ops = metadata.operations;
    expect(ops[0].association.api_name).toBe("OpenTarget API");
    expect(ops[0].association.predicate).toBe("related_to");
    expect(ops[0].association.input_id).toBe("ENSEMBL");
    expect(ops[0].query_operation.path).toBe("/platform/public/evidence/filter");
  });
});

describe("test API parser using specs with path parameters", () => {
  test("test path params", () => {
    const smartapi_spec = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, "../../data/litvar.json"), { encoding: "utf8" }),
    ) as SmartAPISpec;
    const litvar = new API(smartapi_spec);
    const path_params = litvar.metadata.operations[0].query_operation?.path_params;
    expect(path_params).toStrictEqual(["variantid"]);
  });
});
