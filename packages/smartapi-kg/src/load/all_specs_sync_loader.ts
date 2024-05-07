import BaseLoader from "./base_loader";
import { SmartAPIQueryResult } from "../types";
import { SmartAPISpec } from "../parser/types";
import fs from "fs";
import Debug from "debug";
const debug = Debug("retriever:smartapi-kg:AllSpecsSyncLoader");

export default class AllSpecsSyncLoader extends BaseLoader {
  private _file_path: string;
  constructor(path: string) {
    super();
    this._file_path = path;
  }
  protected fetch(): SmartAPIQueryResult {
    debug(`Fetching from file path: ${this._file_path}`);
    const file = fs.readFileSync(this._file_path, "utf-8");
    const data = JSON.parse(file) as SmartAPIQueryResult | SmartAPISpec;
    let result;
    if (!("hits" in data)) {
      result = {
        hits: [data],
      };
    } else {
      result = data;
    }
    debug(`Hits in inputs: ${"hits" in data}`);
    return result;
  }

  protected parse(input: SmartAPIQueryResult): SmartAPISpec[] {
    return input.hits;
  }

  load(): SmartAPISpec[] {
    const specs = this.fetch();
    return this.parse(specs);
  }
}
