import BaseTransformer from "./transformer";
import { generateFilterString } from "../jq_utils";
import fs from "fs";
import Path from "path";

import * as jq from "node-jq"; // If converted to import, ts compile breaks it (imports default as undefined)
import { JSONDoc, PairedResponse } from "../json_transform/types";
import { toArray } from "../utils";

// Get prefab JQ strings from data/jq
const filterStringsWrap = Object.fromEntries(
  fs.readdirSync(Path.resolve(`${__dirname}/../../data/jq/wrap`)).map(file => {
    const filePath = Path.resolve(`${__dirname}/../../data/jq/wrap/${file}`);
    return [
      Path.parse(filePath).name,
      fs.readFileSync(filePath, { encoding: "utf8" }),
    ];
  }),
);

const filterStringsPair = Object.fromEntries(
  fs.readdirSync(Path.resolve(`${__dirname}/../../data/jq/pair`)).map(file => {
    const filePath = Path.resolve(`${__dirname}/../../data/jq/pair/${file}`);
    return [
      Path.parse(filePath).name,
      fs.readFileSync(filePath, { encoding: "utf8" }),
    ];
  }),
);

export default class JQTransformer extends BaseTransformer {
  // TODO more specific typing?
  async wrap(res: JSONDoc | JSONDoc[]): Promise<JSONDoc> {
    let filterString: string | undefined =
      this.config.wrap ?? filterStringsWrap[this.config.type];
    if (typeof filterString === "undefined") return super.wrap(res);
    filterString = generateFilterString(filterString);
    return JSON.parse(
      (await jq.run(filterString, res, {
        input: "json",
      })) as string,
    );
  }

  async pairCurieWithAPIResponse(): Promise<PairedResponse> {
    let filterString: string | undefined =
      this.config.pair ?? filterStringsPair[this.config.type];
    if (typeof filterString === "undefined")
      return super.pairCurieWithAPIResponse();
    const data = {
      response: this.data.response,
      edge: {
        query: {
          method: this.edge.query_operation.method,
        },
        input: {
          id: this.edge.association.input_id,
          type: this.edge.association.input_type,
          // input is array or is object with queryInputs
          curies:
            Array.isArray(this.edge.input) ||
              typeof this.edge.input === "string"
              ? toArray(this.edge.input)
              : toArray(this.edge.input.queryInputs),
        },
        predicate: this.edge.association.predicate,
        output: {
          id: this.edge.association.output_id,
          type: this.edge.association.output_type,
        },
        response_mapping: this.edge.response_mapping,
      },
    };
    filterString = `.edge as $edge | .response as $response | ${generateFilterString(filterString)}`;
    return JSON.parse(
      (await jq.run(filterString, data, { input: "json" })) as string,
    );
  }
}
