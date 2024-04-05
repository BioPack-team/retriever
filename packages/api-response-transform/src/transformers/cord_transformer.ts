import BioThingsTransformer from "./biothings_transformer";

// deprecated
export default class CordTransformer extends BioThingsTransformer {
  wrap(res) {
    const PREFIXES = ["pr", "go", "mop", "hgnc", "uberon", "so", "cl", "doid", "chebi"];
    let result = {};
    for (let predicate of Object.keys(res)) {
      let tmp = [];
      if (Array.isArray(res[predicate]) && res[predicate].length > 0) {
        res[predicate].map(item => {
          if (
            item["@type"] === this.edge.association.output_type ||
            (item["@type"] === "DiseaseOrPhenotypicFeature" && this.edge.association.output_type === "Disease")
          ) {
            for (let key of Object.keys(item)) {
              if (PREFIXES.includes(key)) {
                item[key.toUpperCase()] = item[key];
                delete item[key];
              }
            }
            tmp.push(item);
          }
        });
      }
      if (tmp.length > 0) {
        result["related_to"] = tmp;
      }
    }
    return result;
  }

  jsonTransform(res) {
    return res;
  }
}
