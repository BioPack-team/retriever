import BaseAsyncLoader from "./base_async_loader";
import { COMPONENT_SMARTAPI_QUERY_TEMPLATE } from "../config";
import { SmartAPIQueryResult } from "../types";
import { SmartAPISpec } from "../parser/types";

export default class ComponentSpecsAsyncLoader extends BaseAsyncLoader {
  constructor(component: string) {
    super(COMPONENT_SMARTAPI_QUERY_TEMPLATE.replace("{component_name}", component));
  }

  protected async fetch(): Promise<SmartAPIQueryResult> {
    return super.fetch() as Promise<SmartAPIQueryResult>;
  }

  protected parse(input: SmartAPIQueryResult): SmartAPISpec[] {
    return input.hits;
  }
}
