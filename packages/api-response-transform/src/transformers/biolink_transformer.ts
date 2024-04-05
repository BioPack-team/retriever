import BaseTransformer from "./transformer";

export default class BiolinkTransformer extends BaseTransformer {
  wrap(res) {
    //super.wrap();
    const PREFIXES = ["HGNC", "NCBIGene", "REACT"];
    if ("associations" in res) {
      for (let rec of res.associations) {
        if (rec.object && "id" in rec.object) {
          let prefix, value;
          [prefix, value] = rec.object.id.split(":");
          if (PREFIXES.includes(prefix)) {
            rec["object"][prefix] = value;
          } else {
            rec["object"][prefix] = rec.object.id;
          }
        }
        if (rec.publications === undefined || rec.publications.length === 0) {
          delete rec.publications;
        } else {
          const oldPublications = rec.publications;
          rec.publications = [];
          for (let oldPub of oldPublications) {
            if (!oldPub?.id?.startsWith?.("PMID:")) {
              continue;
            }

            rec.publications.push({ id: oldPub.id.split(":").slice(-1)[0] });
          }
        }
        if (!("provided_by" in rec)) {
          delete rec.provided_by;
        }
      }
    }
    return res;
  }
}
