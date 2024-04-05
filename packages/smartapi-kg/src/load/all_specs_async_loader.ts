import BaseAsyncLoader from "./base_async_loader";
import { SMARTAPI_URL } from "../config";
import { SmartAPIQueryResult } from "../types";
import { SmartAPISpec } from "../parser/types";

export default class AllSpecsAsyncLoader extends BaseAsyncLoader {
  constructor() {
    super(SMARTAPI_URL);
  }

  protected async fetch(): Promise<SmartAPIQueryResult> {
    return super.fetch() as Promise<SmartAPIQueryResult>;
  }

  protected parse(input: SmartAPIQueryResult): SmartAPISpec[] {
    return input.hits;
  }
}
