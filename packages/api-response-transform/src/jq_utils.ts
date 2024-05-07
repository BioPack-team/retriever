import { JQVariable } from "./types";
import Path from "path";
import fs from "fs";

const functions = fs.readFileSync(
  Path.resolve(`${__dirname}/../data/jq/utils.jq`),
  { encoding: "utf8" },
);

function generateVariables(variables: JQVariable[]) {
  let variableString = "";
  for (const variable of variables) {
    variableString += `${variable.value} as ${variable.name} | `;
  }
  return variableString;
}

export function generateFilterString(filterString: string) {
  return `${functions}\n${filterString}`;
}
