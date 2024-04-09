import { APIList } from "@biothings-explorer/types";
import { readFileSync } from "fs";
import yaml from "js-yaml";
import path = require("path");

let API_LIST: APIList;

if (!API_LIST) {
  API_LIST = yaml.load(
    readFileSync(path.resolve(__dirname, "../../config/api_list.yaml"), { encoding: "utf8" }),
  ) as APIList;
}

export default API_LIST;
