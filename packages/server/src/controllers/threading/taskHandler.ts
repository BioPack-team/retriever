import "../opentelemetry";

import { isMainThread, threadId } from "worker_threads";
import Piscina from "piscina";
import Debug from "debug";
const debug = Debug(`bte:biothings-explorer-trapi:worker${threadId}`);

if (!isMainThread) {
  // Log thread start before BioLink model loads
  debug(`Worker thread ${threadId} is ready to accept ${Piscina.workerData.queue} tasks.`);
}

import { tasks } from "../../routes/index";
import { getQueryQueue } from "../async/asyncquery_queue";
import * as Sentry from "@sentry/node";
import { ProfilingIntegration } from "@sentry/profiling-node";
import OpenTelemetry, { Span } from "@opentelemetry/api";
import { Telemetry } from "@biothings-explorer/utils";
import { InnerTaskData } from "@biothings-explorer/types";

// use SENTRY_DSN environment variable
try {
  Sentry.init({
    integrations: [
      // Automatically instrument Node.js libraries and frameworks
      ...Sentry.autoDiscoverNodePerformanceMonitoringIntegrations(),
      new ProfilingIntegration(),
      // enable HTTP calls tracing
      new Sentry.Integrations.Http({ tracing: true }),
    ],
    environment: process.env.INSTANCE_ENV,
    debug: true,
    normalizeDepth: 6,
    maxBreadcrumbs: 500,
    // Set tracesSampleRate to 1.0 to capture 100%
    // of transactions for performance monitoring.
    // We recommend adjusting this value in production
    tracesSampleRate: process.env.THREAD_SAMPLE_RATE ? parseFloat(process.env.THREAD_SAMPLE_RATE) : 1.0,
    profilesSampleRate: process.env.THREAD_PROFILE_RATE ? parseFloat(process.env.THREAD_PROFILE_RATE) : 1.0, // Profiling sample rate is relative to tracesSampleRate,
    _experiments: {
      maxProfileDurationMs: 6 * 60 * 1000, // max profiling duration of 6 minutes (technically "beta" feature)
    },
  });
} catch (error) {
  debug("Sentry init error. This does not affect execution.");
  debug(error);
}

async function runTask({
  req,
  route,
  traceparent,
  tracestate,
  port,
  job = { jobId: undefined, queueName: undefined },
}: InnerTaskData) {
  debug(`Worker thread ${threadId} beginning ${Piscina.workerData.queue} task.`);

  global.SCHEMA_VERSION = "1.4.0";

  global.parentPort = port;
  port.postMessage({ threadId, registerId: true });
  global.cachingTasks = [];

  global.queryInformation = {
    queryGraph: req?.data?.queryGraph,
  };

  if (job.queueName) {
    const queue = getQueryQueue(job.queueName);
    global.job = await queue.getJob(job.jobId);
  }

  const routeNames = {
    query_v1: "EXEC /v1/query",
    query_v1_by_api: "EXEC /v1/smartapi/:/query",
    query_v1_by_team: "EXEC /v1/team/:/query",
    asyncquery_status: "EXEC /v1/asyncquery_status",
    asyncquery_v1: "EXEC /v1/asyncquery",
    asyncquery_v1_by_api: "EXEC /v1/smartapi/:/asyncquery",
    asyncquery_v1_by_team: "EXEC /v1/team/:/asyncquery",
  };

  let transaction: Sentry.Transaction, span: Span;
  try {
    transaction = Sentry.startTransaction({ name: routeNames[route] });
    transaction.setData("request", req.data.queryGraph);
    Sentry.getCurrentHub().configureScope(scope => {
      scope.clearBreadcrumbs();
      scope.setSpan(transaction);
    });

    span = OpenTelemetry.trace
      .getTracer("biothings-explorer-thread")
      .startSpan(
        routeNames[route],
        undefined,
        OpenTelemetry.propagation.extract(OpenTelemetry.context.active(), { traceparent, tracestate }),
      );
    span.setAttribute("bte.requestData", JSON.stringify(req.data.queryGraph));
    Telemetry.setOtelSpan(span);
  } catch (error) {
    debug("Sentry/OpenTelemetry transaction start error. This does not affect execution.");
    debug(error);
  }

  const completedTask = await tasks[route](req);
  await Promise.all(global.cachingTasks);

  try {
    transaction.finish();
    span.end();
    Telemetry.removeOtelSpan();
  } catch (error) {
    debug("Sentry/OpenTelemetry transaction finish error. This does not affect execution.");
    debug(error);
  }

  debug(`Worker thread ${threadId} completed ${Piscina.workerData.queue} task.`);

  return completedTask;
}

module.exports = runTask;
