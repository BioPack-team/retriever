import SingleSpecAsyncLoader from "./single_spec_async_loader";
import TeamSpecsAsyncLoader from "./team_specs_async_loader";
import AllSpecsAsyncLoader from "./all_specs_async_loader";
import TagSpecsAsyncLoader from "./tag_specs_async_loader";
import ComponentSpecsAsyncLoader from "./component_specs_async_loader";
import { SmartAPISpec } from "../parser/types";

export const asyncLoaderFactory = async (
  smartAPIID: string = undefined,
  teamName: string = undefined,
  tag: string = undefined,
  component: string = undefined,
): Promise<SmartAPISpec[]> => {
  let loader;
  if (!(typeof smartAPIID === "undefined")) {
    loader = new SingleSpecAsyncLoader(smartAPIID);
  } else if (!(typeof teamName === "undefined")) {
    loader = new TeamSpecsAsyncLoader(teamName);
  } else if (!(typeof tag === "undefined")) {
    loader = new TagSpecsAsyncLoader(tag);
  } else if (!(typeof component === "undefined")) {
    loader = new ComponentSpecsAsyncLoader(component);
  } else {
    loader = new AllSpecsAsyncLoader();
  }
  const specs = await loader.load();
  return specs;
};
