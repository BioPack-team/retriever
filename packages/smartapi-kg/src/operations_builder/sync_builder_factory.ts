import SyncOperationsBuilder from "./sync_operations_builder";
import SyncOperationsBuilderWithReasoner from "./sync_operations_builder_with_reasoner";
import { BuilderOptions } from "../types";
import { SmartAPIKGOperationObject, SmartAPISpec } from "../parser/types";

export function syncBuilderFactory(
  options: BuilderOptions,
  includeReasoner: boolean,
  smartapi_path: string,
  predicates_path: string,
): SmartAPIKGOperationObject[] {
  let builder;
  if (includeReasoner === true) {
    builder = new SyncOperationsBuilderWithReasoner(options, smartapi_path, predicates_path);
  } else {
    builder = new SyncOperationsBuilder(options, smartapi_path);
  }
  const ops = builder.build();

  const primaryKnowledgeAPIs = new Set();
  options.apiList?.include.forEach(api => {
    if (api.primarySource) {
      if (api.id) primaryKnowledgeAPIs.add(api.id);
      if (api.infores) primaryKnowledgeAPIs.add(api.infores);
    }
  });
  options.apiList?.exclude.forEach(api => {
    if (api.primarySource) {
      if (api.id) primaryKnowledgeAPIs.add(api.id);
      if (api.infores) primaryKnowledgeAPIs.add(api.infores);
    }
  });
  ops.map(op => {
    const apiIsPrimaryKnowledgeSource =
      primaryKnowledgeAPIs.has(op.association.smartapi.id) ||
      primaryKnowledgeAPIs.has(op.association["x-translator"].infores);
    op.association.apiIsPrimaryKnowledgeSource = apiIsPrimaryKnowledgeSource ? true : false;
  });
  return ops;
}
