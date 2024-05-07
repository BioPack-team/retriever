// import {
//   NodeNormalizerResultObj,
//   Record,
// } from "@retriever/api-response-transform";
// import { LogEntry } from "@retriever/utils";
// import SubqueryDispatcher from "./dispatcher";
// import { JSONDoc, QueryHandlerOptions, UnavailableAPITracker } from "./types";
// import Subquery from "./queries/subquery";
// import Debug from "debug";
// import {
//   ResolvableBioEntity,
//   ResolverOutput,
//   SRIResolverOutput,
//   generateInvalidBioentities,
//   getAttributes,
//   resolveSRI,
// } from "biomedical_id_resolver";
// const debug = Debug("retriever:call-apis:query");
// import ESSerializer from 'esserializer';
// import TemplateSubquery from "./queries/template_subquery";
// import TrapiSubquery from "./queries/trapi_subquery";
// import { QEdge } from "@retriever/graph";
//
// ESSerializer.registerClasses([
//   Subquery,
//   TemplateSubquery,
//   TrapiSubquery,
//   QEdge,
// ]);
//
// const subqueryDispatcher = new SubqueryDispatcher();
//
//
//
// export default async function query({
//   query,
//   resolveOutputIDs = true,
//   options,
// }: {
//   query: string;
//   resolveOutputIDs: boolean;
//   options: QueryHandlerOptions;
// }) {
//   // let startTime = performance.now();
//
//   const { records, logs } = await subqueryDispatcher.execute(ESSerializer.deserialize(query), options);
//
//   // let finishTime = performance.now();
//   // const timeElapsed = Math.round(
//   //   finishTime - startTime > 1000
//   //     ? (finishTime - startTime) / 1000
//   //     : finishTime - startTime,
//   // );
//   // // const timeUnits = finishTime - startTime > 1000 ? "s" : "ms";
//   debug("query completes.");
//
//   debug("Start to use id resolver module to annotate output ids.");
//   const annotatedRecords = await annotate(records, resolveOutputIDs);
//   debug("id annotation completes");
//
//   return { records: Record.packRecords(annotatedRecords), logs };
// }
