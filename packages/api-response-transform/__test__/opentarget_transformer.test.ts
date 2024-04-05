import jq_tf from "../built/transformers/jq_transformer";
import fs from "fs";
import path from "path";

describe("test opentarget transformer", () => {
  let response;
  let input;

  beforeEach(() => {
    const response_path = path.resolve(__dirname, "./data/opentarget/response.json");
    response = JSON.parse(fs.readFileSync(response_path, { encoding: "utf8" }));
    const edge_path = path.resolve(__dirname, "./data/opentarget/edge.json");
    const edge = JSON.parse(fs.readFileSync(edge_path, { encoding: "utf8" }));
    input = {
      response,
      edge,
    };
  });

  // skip these tests since we're not ingesting opentargets right now
  test.skip("test opentarget wrapper", async () => {
    const tf = new jq_tf(input, { type: "opentarget" });
    const res = await tf.wrap(response);
    expect(res).toHaveProperty("data");
    expect(res.data[0].drug.id).toEqual("CHEMBL220492");
  });

  test.skip("test opentarget wrapper if id field is not chembl", async () => {
    const tf = new jq_tf(input, { type: "opentarget" });
    const fake = {
      data: [
        {
          drug: {
            id: "http://identifiers.org/drugbank/DB0001",
          },
        },
      ],
    };
    const res = await tf.wrap(fake);
    expect(res).toHaveProperty("data");
    expect(res.data[0].drug.id).toEqual("http://identifiers.org/drugbank/DB0001");
  });
});
