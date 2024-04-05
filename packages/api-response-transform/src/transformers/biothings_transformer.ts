import BaseTransformer from "./transformer";
import { generateCurie } from "../utils";

export default class BioThingsTransformer extends BaseTransformer {
  async pairCurieWithAPIResponse() {
    if (this.edge.query_operation.method === "post") {
      let res = {};
      const mapper = item => {
        // for input not found, BioThings API returns an entry with a key "notfound" equal to true
        if (!("notfound" in item)) {
          let input = generateCurie(this.edge.association.input_id, item.query);
          if (input in res) {
            res[input].push(item);
          } else {
            res[input] = [item];
          }
        }
      };

      if (this.data.response.hasOwnProperty("hits")) {
        // @ts-ignore
        this.data.response.hits.map(mapper);
      } else {
        // @ts-ignore
        this.data.response.map(mapper);
      }
      return res;
    } else {
      return super.pairCurieWithAPIResponse();
    }
  }
}
