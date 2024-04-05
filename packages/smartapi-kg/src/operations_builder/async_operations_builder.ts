import BaseOperationsBuilder from "./base_operations_builder";
import { asyncLoaderFactory } from "../load/async_loader_factory";
import { BuilderOptions } from "../types";
import { SmartAPISpec } from "../parser/types";

export default class ASyncOperationsBuilder extends BaseOperationsBuilder {
  constructor(options: BuilderOptions) {
    super(options);
  }

  protected async load(): Promise<SmartAPISpec[]> {
    const specs = await asyncLoaderFactory(
      this._options.smartAPIID,
      this._options.teamName,
      this._options.tag,
      this._options.component,
    );
    return specs;
  }

  async build() {
    const specs = await this.load();
    return this.loadOpsFromSpecs(specs);
  }
}
