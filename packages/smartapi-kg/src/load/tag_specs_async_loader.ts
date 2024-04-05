import BaseAsyncLoader from "./base_async_loader";
import { TAG_SMARTAPI_QUERY_TEMPLATE } from "../config";
import { SmartAPIQueryResult } from "../types";
import { SmartAPISpec } from "../parser/types";

export default class TagSpecsAsyncLoader extends BaseAsyncLoader {
  constructor(tag: string) {
    super(TAG_SMARTAPI_QUERY_TEMPLATE.replace("{tag_name}", tag));
  }

  protected async fetch(): Promise<SmartAPIQueryResult> {
    return super.fetch() as Promise<SmartAPIQueryResult>;
  }

  protected parse(input: SmartAPIQueryResult): SmartAPISpec[] {
    return input.hits;
  }
}
