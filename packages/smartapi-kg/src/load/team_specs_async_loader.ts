import BaseAsyncLoader from "./base_async_loader";
import { TEAM_SMARTAPI_QUERY_TEMPLATE } from "../config";
import { SmartAPIQueryResult } from "../types";
import { SmartAPISpec } from "../parser/types";

export default class TeamSpecsAsyncLoader extends BaseAsyncLoader {
  constructor(teamName: string) {
    super(TEAM_SMARTAPI_QUERY_TEMPLATE.replace("{team_name}", teamName));
  }

  protected async fetch(): Promise<SmartAPIQueryResult> {
    return super.fetch() as Promise<SmartAPIQueryResult>;
  }

  protected parse(input: SmartAPIQueryResult): SmartAPISpec[] {
    return input.hits;
  }
}
