import AllSpecsSyncLoader from "./all_specs_sync_loader";
import { SmartAPIQueryResult, apiListObject } from "../types";
import { SmartAPISpec } from "../parser/types";
import Debug from "debug";
const debug = Debug("bte:smartapi-kg:APIListSpecsSyncLoader");

export default class APIListSpecsSyncLoader extends AllSpecsSyncLoader {
  private _apiList: apiListObject | undefined;

  constructor(path: string, apiList?: apiListObject) {
    super(path);
    this._apiList = apiList;
  }

  parse(input: SmartAPIQueryResult): SmartAPISpec[] {
    if (!this._apiList) {
      return input.hits;
    }
    return input.hits.filter(item => {
      const includeSmartAPI = this._apiList?.include.find(api => api.id === item._id && api.id !== undefined);
      const includeInfoRes = this._apiList?.include.find(
        api => api.infores === item.info["x-translator"]?.infores && api.infores !== undefined,
      );
      const excludeSmartAPI = this._apiList?.exclude.find(api => api.id === item._id && api.id !== undefined);
      const excludeInfoRes = this._apiList?.exclude.find(
        api => api.infores === item.info["x-translator"]?.infores && api.infores !== undefined,
      );

      let willBeIncluded;
      let apiValue;
      if (excludeSmartAPI) {
        willBeIncluded = false;
        apiValue = excludeSmartAPI;
      } else if (includeSmartAPI) {
        willBeIncluded = true;
        apiValue = includeSmartAPI;
      } else if (excludeInfoRes) {
        willBeIncluded = false;
        apiValue = excludeInfoRes;
      } else if (includeInfoRes) {
        apiValue = includeInfoRes;
        willBeIncluded = true;
      } else {
        apiValue = undefined;
        willBeIncluded = false;
      }

      if (apiValue && apiValue.name !== item.info.title) {
        debug(
          `Expected to get '${apiValue.name}' with smartapi-id:${apiValue.id} but instead got '${item.info.title}'`,
        );
      }

      return willBeIncluded;
    });
    // .filter(item => {
    //   let api = this._apiList.exclude.find(api => api.id === item._id);
    //   if (api && api.name !== item.info.title) {
    //     debug(`Expected to get '${api.name}' with smartapi-id:${api.id} but instead got '${item.info.title}'`);
    //   }
    //   return !api;
    // });
  }
}
