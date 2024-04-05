import { SmartAPIQueryResult, apiListObject } from "../types";
import { SmartAPISpec } from "../parser/types";
import APIListSpecsSyncLoader from "./api_list_specs_sync_loader";

export default class TeamSpecsSyncLoader extends APIListSpecsSyncLoader {
  private _teamName: string;

  constructor(teamName: string, path: string, apiList?: apiListObject) {
    super(path, apiList);
    this._teamName = teamName;
  }

  parse(input: SmartAPIQueryResult): SmartAPISpec[] {
    return super
      .parse(input)
      .filter(
        item =>
          "x-translator" in item.info &&
          "team" in item.info["x-translator"] &&
          Array.isArray(item.info["x-translator"].team) &&
          item.info["x-translator"].team.includes(this._teamName),
      );
  }
}
