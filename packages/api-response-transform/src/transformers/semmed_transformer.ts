import BioThingsTransformer from "./biothings_transformer";

export default class SemmedTransformer extends BioThingsTransformer {
  wrap(res) {
    let result = {};
    for (let predicate of Object.keys(res)) {
      let tmp = [];
      if (Array.isArray(res[predicate]) && res[predicate].length > 0) {
        res[predicate].map(item => {
          if (
            item["@type"] === this.edge.association.output_type ||
            (item["@type"] === "DiseaseOrPhenotypicFeature" && this.edge.association.output_type === "Disease")
          ) {
            item["UMLS"] = item["umls"];
            item["pubmed"] = item["pmid"];
            delete item["umls"];
            delete item["pmid"];
            tmp.push(item);
          }
        });
      }
      if (tmp.length > 0) {
        result[predicate] = tmp;
      }
    }
    return result;
  }

  jsonTransform(res) {
    return res;
  }
}
