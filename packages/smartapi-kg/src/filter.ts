import { FILTER_FIELDS } from "./config";
import _ from "lodash";
import { FilterCriteria, ObjectWithValueAsSet, CompactQualifiers } from "./types";
import { SmartAPIKGOperationObject } from "./parser/types";
// import Debug from "debug";
// const debug = Debug("bte:smartapi-kg:Filter");

const getUniqueValsForEachField = (operations: SmartAPIKGOperationObject[]): ObjectWithValueAsSet => {
  const allValues = {} as ObjectWithValueAsSet;
  FILTER_FIELDS.forEach(field => (allValues[field] = new Set()));
  const qualifiers: { [key: string]: any } = {};
  operations.map(operation => {
    FILTER_FIELDS.map(field => {
      if (field === "component") {
        //component is the only nested field
        allValues[field].add(operation.association["x-translator"][field]);
      } else if (field === "qualifiers") {
        if (operation.association[field] === undefined) {
          return;
        }
        const key: string = Object.entries(operation.association[field]).reduce((str, [key, value]) => {
          return str + key + value;
        }, "");
        qualifiers[key] = operation.association[field];
      } else {
        allValues[field].add(operation.association[field]);
      }
    });
  });
  allValues["qualifiers"] = new Set(Object.values(qualifiers));
  return allValues;
};

/**
 * filter an array of objects based on the filter criteria
 * @param {Array} ops - an array of objects, each represents an association
 * @param {Object} criteria - the filter criteria
 */
export function ft(ops: SmartAPIKGOperationObject[], criteria: FilterCriteria): SmartAPIKGOperationObject[] {
  const allValues = getUniqueValsForEachField(ops);
  const filters = {} as ObjectWithValueAsSet;

  FILTER_FIELDS.map(field => {
    if (!(field in criteria) || criteria[field] === undefined) {
      filters[field] = allValues[field];
    } else {
      const vals = Array.isArray(criteria[field])
        ? (criteria[field] as (string | CompactQualifiers[])[])
        : [criteria[field]];
      filters[field] = new Set(vals);
    }
  });

  let res = _.cloneDeep(ops);
  FILTER_FIELDS.forEach(field => {
    res = res.filter(rec => {
      if (field === "component") {
        // component is the only nested field
        return filters[field].has(rec.association["x-translator"][field]) ? true : false;
      } else if (field === "qualifiers") {
        // return true;
        if (criteria[field] === undefined || criteria[field].length < 1) {
          return true;
        }
        return [...filters[field]].some(qualifierConstraintSet => {
          return Object.entries(qualifierConstraintSet).every(
            ([qualifierType, qualifierValue]: [qualifierType: string, qualifierValue: string | string[]]) => {
              if (!rec.association[field]) return false;
              const qualifierValueArray = Array.isArray(qualifierValue) ? qualifierValue : [qualifierValue];
              const associationValueArray = Array.isArray(rec.association[field][qualifierType])
                ? rec.association[field][qualifierType]
                : [rec.association[field][qualifierType]];

              return qualifierValueArray.some(value => {
                return associationValueArray.includes(value);
              });
            },
          );
        });
      } else {
        return filters[field].has(rec.association[field]) ? true : false;
      }
    });
  });
  return res;
}
