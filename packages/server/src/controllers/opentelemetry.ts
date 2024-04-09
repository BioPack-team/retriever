import { NodeSDK } from "@opentelemetry/sdk-node";
import { getNodeAutoInstrumentations } from "@opentelemetry/auto-instrumentations-node";
import { Resource } from "@opentelemetry/resources";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer:otel-init");
import { JaegerExporter } from "@opentelemetry/exporter-jaeger";

debug("Initializing Opentelemetry instrumentation...");
const sdk = new NodeSDK({
  traceExporter: new JaegerExporter({
    host: process.env.JAEGER_HOST ?? "jaeger-otel-agent.sri",
    port: parseInt(process.env.JAEGER_PORT ?? "6832"),
  }),
  instrumentations: [getNodeAutoInstrumentations()],
  resource: new Resource({
    "service.name": "biothings-explorer",
  }),
});
sdk.start();
debug("Opentelemetry instrumentation initialized.");
