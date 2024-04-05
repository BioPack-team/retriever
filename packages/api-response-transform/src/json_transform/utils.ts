import jsonata from "jsonata";
import _ from "lodash";
import commonpath from "common-path-prefix";
import { BaseTemplate, ComplexTemplate, UserInputTemplate, BaseTransformedObject, JSONDoc } from "./types";
/**
 * Extract all paths from a template object
 * @param {object} template - part of the template
 * @returns {object} - an array of paths
 */
export const extractPathsFromTemplate = (template: BaseTemplate): string[] => {
  let paths: string[] = [];
  Object.values(template).map(val => {
    if (typeof val === "string") {
      paths.push(val);
    } else {
      paths = [...paths, ...val];
    }
  });
  return paths;
};

/**
 * find the longest common path given an array of paths
 * @param {object} paths - an array of paths (e.g. go.BP.id)
 * @returns - the longest common path(separated by dot)
 */
export const findLongestCommonPath = (paths: string[]): string => {
  if (paths.length === 0) {
    return undefined;
  }
  if (paths.length === 1) {
    return paths[0];
  }
  const common_path = commonpath(paths, ".");
  return common_path.endsWith(".") ? common_path.slice(0, -1) : undefined;
};

/**
 * Transform a simple JSON object based on the template
 * @param {object} json_doc - the JSON object to be transformed
 * @param {object} template - the template on which the transform is based
 * @returns {object} - the transformed json object
 */
export const transformSimpleObject = (json_doc: JSONDoc, template: BaseTemplate): BaseTransformedObject => {
  const new_doc = {} as BaseTransformedObject;
  let val;
  let expression;
  if (Object.keys(json_doc).length === 0) {
    return new_doc;
  }
  for (let [key, value] of Object.entries(template)) {
    if (typeof value === "string") {
      expression = jsonata(value);
      val = [expression.evaluate(json_doc)];
    } else {
      val = value.map(element => jsonata(element).evaluate(json_doc));
    }
    val = val.filter(item => !(typeof item === "undefined"));
    if (val.length === 0) continue;
    if (val.length === 1) val = val[0];
    new_doc[key] = val;
  }
  return new_doc;
};

/**
 * Transform an array of simple JSON object based on the template
 * @param {array} json_doc - the JSON object to be transformed
 * @param {object} template - the template on which the transform is based
 * @returns {array} - the transformed json object
 */
export const transformArrayOfSimpleObject = (
  json_doc: JSONDoc | JSONDoc[],
  template: BaseTemplate,
): BaseTransformedObject | BaseTransformedObject[] => {
  if (Array.isArray(json_doc)) {
    return json_doc.map(_doc => transformSimpleObject(_doc, template));
  }
  return json_doc;
};

/**
 * Transform a complex JSON object based on the template
 * @param {object} json_doc - the JSON object to be transformed
 * @param {object} template - the template on which the transform is based
 * @returns {object} - the transformed json object
 */
export const transformComplexObject = (json_doc: JSONDoc, template: BaseTemplate) => {
  let new_doc = {};
  let trimmed_json_doc;
  let trimmed_template;
  const paths = extractPathsFromTemplate(template);
  const common_path = findLongestCommonPath(paths);
  if (paths.length === 1 && paths[0] === common_path) {
    return transformSimpleObject(json_doc, template);
  }
  if (common_path) {
    trimmed_json_doc = jsonata(common_path).evaluate(json_doc);
    trimmed_template = removeCommonPathFromTemplate(template, common_path);
  } else {
    trimmed_json_doc = json_doc;
    trimmed_template = template;
  }
  if (typeof trimmed_json_doc === "undefined") {
    return {};
  }
  if (Array.isArray(trimmed_json_doc)) {
    new_doc = transformArrayOfSimpleObject(trimmed_json_doc, trimmed_template);
  } else {
    new_doc = transformSimpleObject(trimmed_json_doc, trimmed_template);
  }
  return new_doc;
};

/**
 * Remove common prefix from template
 * @param {object} template - part of the template on which the transform is based
 * @param {string} common_path - the longest common path which all keys in the template share
 * @returns {object} - new template of which the common path has been removed
 */
export const removeCommonPathFromTemplate = (template: BaseTemplate, common_path: string): BaseTemplate => {
  if (typeof common_path !== "string") {
    return template;
  }
  common_path = common_path + ".";
  const new_template = {} as BaseTemplate;
  for (let [key, value] of Object.entries(template)) {
    if (typeof value === "string") {
      new_template[key] = value.startsWith(common_path) ? value.substring(common_path.length) : value;
    } else {
      const new_value = [] as string[];
      value.map(path => {
        const trimmed_path = path.startsWith(common_path) ? path.substring(common_path.length) : path;
        new_value.push(trimmed_path);
      });
      new_template[key] = new_value;
    }
  }
  return new_template;
};

const isArrayOfString = (input: string | string[] | BaseTemplate): input is string[] => {
  return Array.isArray(input) && input.every(i => typeof i === "string");
};

/**
 * distinguish simple paths and complex paths from template
 * @param {object} template - the template which transformation is based on
 */
export const separateSimpleAnComplexPaths = (template: UserInputTemplate) => {
  const simple_path_template = {} as BaseTemplate;
  const complex_path_template = {} as ComplexTemplate;
  for (let [key, value] of Object.entries(template)) {
    if (typeof value === "string" || isArrayOfString(value)) {
      simple_path_template[key] = value;
    } else {
      complex_path_template[key] = value;
    }
  }
  return { simple_path_template, complex_path_template };
};
