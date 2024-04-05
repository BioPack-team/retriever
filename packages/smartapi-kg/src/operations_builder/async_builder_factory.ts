import ASyncOperationsBuilder from "./async_operations_builder";
import AsyncOperationsBuilderWithReasoner from "./async_operations_builder_with_reasoner";
import { BuilderOptions } from "../types";
import { SmartAPIKGOperationObject, SmartAPISpec } from "../parser/types";

export const asyncBuilderFactory = async (
  options: BuilderOptions,
  includeReasoner: boolean,
): Promise<SmartAPIKGOperationObject[]> => {
  let builder;
  if (includeReasoner === true) {
    builder = new AsyncOperationsBuilderWithReasoner(options);
  } else {
    builder = new ASyncOperationsBuilder(options);
  }
  const ops = await builder.build();
  return ops;
};
