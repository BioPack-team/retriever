import SingleSpecSyncLoader from "./single_spec_sync_loader";
import TeamSpecsSyncLoader from "./team_specs_sync_loader";
import AllSpecsSyncLoader from "./all_specs_sync_loader";
import TagSpecsSyncLoader from "./tag_specs_sync_loader";
import ComponentSpecsSyncLoader from "./component_specs_sync_loader";
import APIListSpecsSyncLoader from "./api_list_specs_sync_loader";
import { SmartAPISpec } from "../parser/types";
import { apiListObject } from "../types";
import Debug from "debug";
const debug = Debug("bte:smartapi-kg:SyncLoader");

export const syncLoaderFactory = (
  smartAPIID: string = undefined,
  teamName: string = undefined,
  tag: string = undefined,
  component: string = undefined,
  apiList: apiListObject = undefined,
  path: string,
): SmartAPISpec[] => {
  let loader;
  if (!(typeof smartAPIID === "undefined")) {
    loader = new SingleSpecSyncLoader(smartAPIID, path, apiList);
    debug("Using single spec sync loader now.");
  } else if (!(typeof teamName === "undefined")) {
    loader = new TeamSpecsSyncLoader(teamName, path, apiList);
    debug("Using team spec sync loader now.");
  } else if (!(typeof tag === "undefined")) {
    loader = new TagSpecsSyncLoader(tag, path);
    debug("Using tags spec sync loader now.");
  } else if (!(typeof component === "undefined")) {
    loader = new ComponentSpecsSyncLoader(component, path);
    debug("Using component spec sync loader now.");
  } else if (!(typeof apiList === "undefined")) {
    loader = new APIListSpecsSyncLoader(path, apiList);
    debug("Using api list spec sync loader now.");
  } else {
    loader = new AllSpecsSyncLoader(path);
    debug("Using all specs sync loader now.");
  }
  return loader.load();
};
