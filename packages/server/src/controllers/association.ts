import meta_kg from "@biothings-explorer/smartapi-kg";
import fs from "fs";
import path from "path";
import util from "util";
const readFile = util.promisify(fs.readFile);
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:metakg");

export interface AssocResult {
  subject: string;
  object: string;
  predicate: string;
  provided_by: string;
  api: {
    name: string;
    smartapi: {
      metadata: string;
      id: string;
      ui: string;
    };
    "x-translator": unknown;
  };
}

export default function (
  sub: string = undefined,
  obj: string = undefined,
  pred: string = undefined,
  component: string = undefined,
  api: string = undefined,
  source: string = undefined,
): AssocResult[] {
  const smartapi_specs = path.resolve(__dirname, "../../data/smartapi_specs.json");
  debug(`smartapi specs loaded: ${smartapi_specs}`);
  const predicates = path.resolve(__dirname, "../../data/predicates.json");
  debug(`predicates endpoints loaded, ${predicates}`);
  const kg = new meta_kg(smartapi_specs, predicates);
  debug("metakg initialized");
  kg.constructMetaKGSync(true, {});
  debug(`metakg loaded: ${kg.ops.length} ops`);
  const associations: AssocResult[] = [];
  const filtered_res = kg.filter({
    input_type: sub,
    output_type: obj,
    predicate: pred,
    api_name: api,
    source: source,
    component: component,
  });
  filtered_res.map(op => {
    associations.push({
      subject: op.association.input_type,
      object: op.association.output_type,
      predicate: op.association.predicate,
      provided_by: op.association.source,
      api: {
        name: op.association.api_name,
        smartapi: {
          metadata: op.association.smartapi.meta.url,
          id: op.association.smartapi.id,
          ui: "https://smart-api.info/ui/" + op.association.smartapi.id,
        },
        "x-translator": op.association["x-translator"],
      },
    });
  });
  return associations;
} // test comment please ignore
