import { SmartAPIQueryResult } from "../types";
import { SmartAPISpec } from "../parser/types";

export default abstract class BaseLoader {
  protected abstract fetch(): SmartAPIQueryResult | SmartAPISpec | Promise<SmartAPIQueryResult | SmartAPISpec>;

  protected abstract parse(input: SmartAPIQueryResult | SmartAPISpec): SmartAPISpec[];

  abstract load(): SmartAPISpec[] | Promise<SmartAPISpec[]>;
}
