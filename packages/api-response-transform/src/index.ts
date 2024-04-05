import BioThingsTransformer from "./transformers/biothings_transformer";
// import CordTransformer from "./transformers/cord_transformer";
import CTDTransformer from "./transformers/ctd_transformer";
import SemmedTransformer from "./transformers/semmed_transformer";
import OpenTargetTransformer from "./transformers/opentarget_transformer";
import BaseTransformer from "./transformers/transformer";
import TRAPITransformer from "./transformers/trapi_transformer";
import EBIProteinTransformer from "./transformers/ebi_protein_transformer";
import JQTransformer from "./transformers/jq_transformer";
import { BTEQueryObject } from "./types";
import { Record } from "./record";
import Debug from "debug";
const debug = Debug("bte:api-response-transform:index");
export * from "./record";
export * from "./types";

export default class Transformer {
  private data: BTEQueryObject;
  private tf: BaseTransformer;
  config: any;
  constructor(data: BTEQueryObject, config: any) {
    this.data = data;
    this.config = config;
    this.route();
  }

  route() {
    const api = this.data.edge.association.api_name;
    debug(`api name ${api}`);
    const tags = this.data.edge.query_operation.tags;
    debug(`api tags: ${tags}`);

    // if (!this.data.edge.query_operation.transformer) {
    //   console.log(`WE DONT DO THE OP ${api}, ${this.data.edge.query_operation}`)
    // }

    if (this.data.edge.query_operation?.transformer?.jq) {
      // console.log("WE DO THE OP", this.data.edge.query_operation.transformer)
      this.tf = new JQTransformer(this.data, {
        ...this.config,
        wrap: this.data.edge.query_operation.transformer.jq.wrap,
        pair: this.data.edge.query_operation.transformer.jq.pair,
      });
    } else if (tags.includes("bte-trapi")) {
      this.tf = new TRAPITransformer(this.data, this.config);
    } else if (api.startsWith("SEMMED")) {
      this.tf = new SemmedTransformer(this.data, this.config);
    } else if (api === "Monarch API") {
      this.tf = new JQTransformer(this.data, { ...this.config, type: "monarch" });
    } else if (api === "EBI Proteins API") {
      // this.tf = new EBIProteinTransformer(this.data, this.config)
      this.tf = new JQTransformer(this.data, { ...this.config, type: "ebi" });
    } else if (tags.includes("biothings")) {
      this.tf = new JQTransformer(this.data, { ...this.config, type: "biothings" });
    } else if (tags.includes("ctd")) {
      this.tf = new JQTransformer(this.data, { ...this.config, type: "ctd" });
    } else if (tags.includes("opentarget")) {
      this.tf = new OpenTargetTransformer(this.data, this.config);
    } else {
      this.tf = new BaseTransformer(this.data, this.config);
    }
  }

  async transform(): Promise<Record[]> {
    return await this.tf.transform();
  }
}
