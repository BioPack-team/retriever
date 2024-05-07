import axios from "axios";
import FailToLoadSpecError from "../exceptions/fail_to_load_spec";
import { SmartAPIQueryResult } from "../types";
import { SmartAPISpec } from "../parser/types";
import BaseLoader from "./base_loader";
import Debug from "debug";
const debug = Debug("retriever:smartapi-kg:AsyncLoader");

export default abstract class BaseAsyncLoader extends BaseLoader {
  protected _url: string;

  constructor(url: string) {
    super();
    this._url = url;
  }

  protected async fetch(): Promise<SmartAPIQueryResult | SmartAPISpec> {
    try {
      const res = await axios
        .get(this._url)
        .then(res => res.data)
        .catch(error => {
          if (error.response) {
            debug(`Query to ${this._url} failed with status code ${error.response.status}`);
            throw new FailToLoadSpecError(`Query to ${this._url} failed with status code ${error.response.status}`);
          }
        });
      return res;
    } catch (e) {
      if (e instanceof FailToLoadSpecError) {
        throw e;
      }
      debug(`Query to ${this._url} failed with error ${e.toString()}`);
      throw new FailToLoadSpecError(`Query to ${this._url} failed with error ${e.toString()}`);
    }
  }

  protected parse(input: SmartAPIQueryResult | SmartAPISpec): SmartAPISpec[] {
    return [];
  }

  async load(): Promise<SmartAPISpec[]> {
    const specs = await this.fetch();
    return this.parse(specs);
  }
}
