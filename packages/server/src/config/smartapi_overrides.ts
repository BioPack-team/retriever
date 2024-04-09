import { SmartApiOverrides } from "../types";
import { readFileSync } from "fs";
import yaml from "js-yaml";
import path = require("path");

export default function getSmartApiOverrideConfig() {
  return yaml.load(
    readFileSync(path.resolve(__dirname, "../../config/smartapi_overrides.yaml"), { encoding: "utf8" }),
  ) as SmartApiOverrides;
}
