import sentry_sdk
from fastapi import FastAPI
from loguru import logger as log
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from sentry_sdk.integrations.opentelemetry import SentryPropagator, SentrySpanProcessor

from retriever.config.general import CONFIG


def configure_telemetry(app: FastAPI | None = None) -> None:
    """Set up both sentry and OTel as configured."""
    if CONFIG.telemetry.sentry_enabled:
        sentry_sdk.init(
            dsn=CONFIG.telemetry.sentry_dsn.get_secret_value()
            if CONFIG.telemetry.sentry_dsn
            else None,
            traces_sample_rate=CONFIG.telemetry.traces_sample_rate,
            profiles_sample_rate=CONFIG.telemetry.profiles_sample_rate,
            instrumenter="otel",
            environment=CONFIG.instance_env,
        )

    if any([CONFIG.telemetry.sentry_enabled, CONFIG.telemetry.otel_enabled]):
        collector_address = ""
        if CONFIG.telemetry.otel_host:
            collector_address = f"http://{CONFIG.telemetry.otel_host.get_secret_value()}:{CONFIG.telemetry.otel_port}"

            log.info(f"Telemetry enabled, settings up service for {collector_address}")

        # Service name is required for most backends
        resource = Resource(attributes={SERVICE_NAME: "retriever"})
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(SentrySpanProcessor())
        # Allows Sentry to still process spans even if there is no OTel collector
        processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{collector_address}{CONFIG.telemetry.otel_trace_endpoint}"
            )
            if CONFIG.telemetry.otel_enabled
            else SpanExporter()  # essentially a NoOp exporter
        )
        trace_provider.add_span_processor(processor)
        trace.set_tracer_provider(trace_provider)
        set_global_textmap(SentryPropagator())

        # Instrument Server
        if app:
            FastAPIInstrumentor.instrument_app(
                app,
                http_capture_headers_server_request=["User-Agent"],
                tracer_provider=trace_provider,
            )

        # Instrument httpx clients
        HTTPXClientInstrumentor().instrument()


def capture_exception(e: Exception) -> None:
    """Capture an otherwise handled exception."""
    span = trace.get_current_span()
    span.record_exception(e, escaped=True)
