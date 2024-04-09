import Debug from "debug";
const debug = Debug("bte:call-apis:query");
import type { APIEdge } from "../types";
import TrapiSubquery from "./trapi_subquery";
import TemplateSubquery from "./template_subquery";
import Subquery from "./subquery";

function subqueryFactory(
  APIEdge: APIEdge,
): Subquery {
  if ("tags" in APIEdge && APIEdge.tags.includes("bte-trapi")) {
    debug(`using trapi builder now`);
    return new TrapiSubquery(APIEdge);
  } else if (APIEdge.query_operation.useTemplating) {
    debug("using template builder");
    return new TemplateSubquery(APIEdge);
  }
  debug("using default builder");
  return new Subquery(APIEdge);
}

export default subqueryFactory;
