import { separateSimpleAnComplexPaths, transformComplexObject, transformSimpleObject } from "./utils";
import { JSONDoc, UserInputTemplate, BaseTemplate, ComplexTemplate } from "./types";

export const transform = (json_doc: JSONDoc | JSONDoc[], template: UserInputTemplate) => {
  let transformed_json_doc;
  let simple_path_template: BaseTemplate;
  let complex_path_template: ComplexTemplate;
  ({ simple_path_template, complex_path_template } = separateSimpleAnComplexPaths(template));
  transformed_json_doc = transformSimpleObject(json_doc, simple_path_template);
  for (let [key, value] of Object.entries(complex_path_template)) {
    if (Array.isArray(value)) {
      transformed_json_doc[key] = value.map(tmpl => transformComplexObject(json_doc, tmpl));
    } else {
      transformed_json_doc[key] = transformComplexObject(json_doc, value);
    }
  }
  return transformed_json_doc;
};
