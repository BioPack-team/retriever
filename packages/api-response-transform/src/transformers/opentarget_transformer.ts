import BaseTransformer from "./transformer";
import jq from "node-jq";

export default class OpenTargetTransformer extends BaseTransformer {
  async wrap(res) {
    const filterString =
      '(.data | .[] | select(.drug.id != null) | select(.drug.id | contains("CHEMBL")) | .drug.id) |= (split("/") | last)';
    return await jq.run(filterString, res);
  }
}
