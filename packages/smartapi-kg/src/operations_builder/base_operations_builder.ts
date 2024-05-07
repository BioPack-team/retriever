import { BuilderOptions } from "../types";
import { SmartAPIKGOperationObject, SmartAPISpec } from "../parser/types";
import API from "../parser/index";
import Debug from "debug";
const debug = Debug("retriever:smartapi-kg:OperationsBuilder");

export default abstract class BaseOperationsBuilder {
  protected _options: BuilderOptions;

  constructor(options: BuilderOptions) {
    this._options = options;
  }

  protected loadOpsFromSpecs(specs: SmartAPISpec[]): SmartAPIKGOperationObject[] {
    let allOps: SmartAPIKGOperationObject[] = [];
    specs.map(spec => {
      try {
        const parser = new API(spec);
        if (!parser.metadata.url) throw new Error("No suitable server present");
        const ops = parser.metadata.operations;
        allOps = [...allOps, ...ops];
      } catch (err) {
        // debug(JSON.stringify(spec.paths))
        debug(`[error]: Unable to parse spec, ${spec ? spec.info.title : spec}. Error message is ${err.toString()}`);
      }
    });
    return allOps;
  }

  abstract build(): SmartAPIKGOperationObject[] | Promise<SmartAPIKGOperationObject[]>;
}
