import * as Sentry from "@sentry/node";
import Debug from "debug";
import opentelemetry, { Span as OtelSpan, Context as OtelContext, SpanStatusCode } from '@opentelemetry/api';
const debug = Debug("bte:telemetry-interface");

const reassurance = "This doesn't affect execution";

class Span {
  span: Sentry.Span;
  otelSpan: OtelSpan;
  constructor(data: unknown) {
    try {
      this.span = Sentry?.getCurrentHub()
        ?.getScope()
        ?.getTransaction()
        ?.startChild(data);

      this.otelSpan = opentelemetry.trace.getTracer('biothings-explorer-thread').startSpan((data as any).description ?? "", undefined, opentelemetry.trace.setSpan(opentelemetry.context.active(), Telemetry.getOtelSpan()));
    } catch (error) {
      debug(`Sentry span start error. ${reassurance}`);
      debug(error);
    }
  }

  setData(key: string, data: unknown) {
    try {
      this.span?.setData(key, data);
      this.otelSpan?.setAttribute(`bte.${key}`, typeof data === 'object' ? JSON.stringify(data) : data as any);
    } catch (error) {
      debug(`Sentry setData error. ${reassurance}`);
      debug(error);
    }
  }

  finish() {
    try {
      this.span?.finish();
      this.otelSpan?.end();
    } catch (error) {
      debug(`Sentry finish error. ${reassurance}`);
      debug(error);
    }
  }
}

export class Telemetry {
  static startSpan(data: unknown) {
    return new Span(data);
  }
  static captureException(error: Error) {
    Sentry.captureException(error);

    if (this.getOtelSpan()) {
        this.getOtelSpan().recordException(error);
        this.getOtelSpan().setStatus({ code: SpanStatusCode.ERROR });
    }
  }
  static addBreadcrumb(breadcrumb?: Sentry.Breadcrumb) {
    try {
      return Sentry?.addBreadcrumb(breadcrumb);
    } catch (error) {
      debug(`Sentry addBreadcrumb error. ${reassurance}`);
      debug(error);
    }
  }
  static setOtelSpan(newOtelSpan: OtelSpan) {
    global.otelSpan = newOtelSpan;
  }
  static getOtelSpan(): OtelSpan {
    return global.otelSpan;
  }
  static removeOtelSpan() {
    global.otelSpan = undefined;
  }
}
