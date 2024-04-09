import { APIDefinition } from "@biothings-explorer/types";
import { readFileSync } from "fs";
import yaml from "js-yaml";
import path = require("path");

let SMARTAPI_EXCLUSIONS: APIDefinition[];

if (!SMARTAPI_EXCLUSIONS) {
  SMARTAPI_EXCLUSIONS = yaml.load(
    readFileSync(path.resolve(__dirname, "../../config/smartapi_exclusions.yaml"), { encoding: "utf8" }),
  ) as APIDefinition[];
}

export default SMARTAPI_EXCLUSIONS;
